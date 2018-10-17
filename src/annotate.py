# pylint: disable=redefined-builtin,redefined-outer-name
"""
Adds entity annotations to JSON-lines files.
"""
import argparse
from collections import deque
import json
import logging
from multiprocessing import JoinableQueue, Lock, Process
import queue
from typing import Any, Deque, Dict, List, Tuple

import spacy
from spacy.tokens import Doc, Token
from sqlitedict import SqliteDict

from src.prefix_tree import PrefixTree
from src.render import process_literal

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


# Custom SpaCy extensions
Doc.set_extension('clusters', default=[])
Token.set_extension('source', default=None)
Token.set_extension('id', default=None)
Token.set_extension('relation', default=None)
Token.set_extension('parent_id', default=None)


def _extract_annotation(token: Token) -> Tuple[str]:
    """Extracts the annotation from a ``Token``."""
    if token._.id is not None:
        annotation = {
            'source': token._.source,
            'id': token._.id,
            'relation': token._.relation,
            'parent_id': token._.parent_id
        }
        return annotation
    return None


def _format_wikilink(wikilink: str) -> str:
    """Formats a wikilink"""
    wikilink = wikilink.replace(' ', '_')
    if len(wikilink) == 1:
        return wikilink.capitalize()
    else:
        return wikilink[0].capitalize() + wikilink[1:]
    return wikilink


def _flatten_tokens(tokens: List[List[str]]) -> List[str]:
    return [word for sent in tokens for word in sent]


class Annotator:
    """Annotates entities within a document by matching aliases to a knowledge
    graph.

    Args:
        alias_db : ``SqliteDict``
            Key-value store mapping an entity's id to a list of aliases for the
            entity.
        relation_db : ``SqliteDict``
            Key-value store mapping an entity's id to a list of related
            entities.
        wiki_db : ``SqliteDict``
            Key-value store mapping wikipedia titles to entity ids.
        distance_cutoff : ``int``
            Maximum distance (in words) allowed between parent and child entities.
    """
    # pylint: disable=too-many-instance-attributes,too-few-public-methods,no-self-use
    def __init__(self,
                 alias_db: SqliteDict,
                 relation_db: SqliteDict,
                 wiki_db: SqliteDict,
                 distance_cutoff: int,
                 match_aliases: bool,
                 unmatch_shady_nel: bool) -> None:
        self._alias_db = alias_db
        self._relation_db = relation_db
        self._wiki_db = wiki_db
        self._distance_cutoff = distance_cutoff
        self._match_aliases = match_aliases
        self._unmatch_shady_nel = unmatch_shady_nel
        self._nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self._last_seen = dict()  # Last location entity was seen in text
        self._parents = dict()  # Lookup parents that added node to graph
        self._alias_lookup = PrefixTree()  # Maps aliases in the text to ids

    def _reset(self):
        """Resets the internal state of the ``Annotator``."""
        self._last_seen = dict()
        self._parents = dict()
        self._alias_lookup = PrefixTree()

    def _json_to_doc(self, json_data: Dict[str, Any]) -> Doc:
        """Converts a JSON object to a SpaCy ``Doc``.

        Args:
            json_data : ``dict``
                Parsed JSON data from an annotation file.

        Returns:
            A SpaCy ``Doc``.
        """
        tokens = _flatten_tokens(json_data['tokens'])
        doc = Doc(self._nlp.vocab, words=tokens)
        # Add wikilink data
        wiki_ids = set()
        for id, start, end in json_data['entities']:
            wiki_ids.add(id)
            if self._match_aliases:
                self._add_aliases(id)
            for token in doc[start:end]:
                token._.id = id
                token._.source = 'WIKI'
        # Add nel data
        SCORE_CUTOFF = 0.5  # TODO: Make this a parameter?
        if 'nel' in json_data:
            for candidate in json_data['nel']:
                if candidate['score'] < SCORE_CUTOFF:
                    continue
                start = candidate['start']
                end = candidate['end']
                already_linked = False
                for token in doc[start:end]:
                    if token._.id is not None:
                        already_linked = True
                if already_linked:
                    continue
                key = _format_wikilink(candidate['label'])
                is_not_shady = key in wiki_ids if self._unmatch_shady_nel else True
                if key in self._wiki_db and is_not_shady:
                    id = self._wiki_db[key]
                    for token in doc[start:end]:
                        token._.id = id
                        token._.source = 'NEL'
        # Add coref clusters
        doc._.clusters = json_data['clusters']
        for _, pipe in self._nlp.pipeline:
            doc = pipe(doc)
        return doc

    def _propagate_ids(self, doc: Doc) -> bool:
        """Propagates id's in coreferent clusters. This may cause
        inconsistencies in previously annotated tokens. To deal with this
        issue, this function also returns a boolean indicating whether or not
        any data was changed.

        Args:
            doc: ``Doc``
                Document to propagate ids in. Modified in place.

        Returns:
            ``True`` if data was changed, otherwise ``False``.
        """
        for i, cluster in enumerate(doc._.clusters):
            logger.debug('Detecting ids for cluster %i', i)
            id_set = set()
            for start, end in cluster:
                cluster_ids = set(token._.id for token in doc[start:end+1] if token._.id)
                if len(cluster_ids) == 1:
                    id_set.add(cluster_ids.pop())
            logger.debug('Id set: %s', id_set)
            if len(id_set) == 1:
                id = id_set.pop()
                logger.debug('Propagating id: %s', id)
                for start, end in cluster:
                    for token in doc[start:end+1]:
                        if token._.id != id:
                            token._.id = id
                            token._.source = 'COREF'
                            logger.debug('token: %s - %s', token, token._)

    def _expand(self, id: str, loc: int) -> None:
        """Expands the session's graph by bringing in related entities/facts.

        Arg:
            id : ``str``
                Identifier for the node to expand.
            loc : ``int``
        """
        logger.debug('Expanding id "%s" at location %i', id, loc)
        if id not in self._last_seen:
            # self._add_aliases(id)
            self._add_relations(id)
        else:
            logger.debug('Node has already been expanded')
        self._last_seen[id] = loc

    def _add_aliases(self, id: str) -> None:
        """Adds aliases for the given node to the alias lookup.

        Args:
            id: ``str``
                Identifier for the node.
        """
        try:
            aliases = self._alias_db[id]
        except KeyError:  # Object is probably a literal
            logger.debug('No aliases found for node "%s"', id)
        else:
            for alias in aliases:
                alias_tokens = [x.text for x in self._nlp.tokenizer(alias)]
                self._alias_lookup.add(alias_tokens, id)

    def _add_relations(self, id: str) -> None:
        """Adds aliases for the given node to the alias lookup.

        Args:
            id: ``str``
                Identifier for the node.
        """
        if id not in self._relation_db:
            logger.debug('Node is not in the relation db!')
            return  # Object is probably a literal
        for prop, value in self._relation_db[id]:
            if value['type'] == 'wikibase-entityid':
                child_id = value['value']['id']
                child_aliases = [] # No longer match non-literals
                # try:
                #     child_aliases = self._alias_db[child_id]
                # except KeyError:  # A child has no name?
                #     logging.warning('Encountered nameless child "%s" in '
                #                     'relation table for "%s"', child_id, id)
            else:
                child_id, child_aliases = process_literal(value)

            if child_id is None:
                continue

            logger.debug('Adding child id "%s" to graph w/ parent "%s"',
                         child_id, id)
            self._parents[child_id] = (prop, id)
            for child_alias in child_aliases:
                child_alias_tokens = [x.text for x in self._nlp.tokenizer(child_alias)]
                if child_alias_tokens not in self._alias_lookup:
                    self._alias_lookup.add(child_alias_tokens, child_id)

    def _existing_id(self,
                     active: Token,
                     token_stack: Deque[Token]) -> Deque[Tuple[Token, str]]:
        """Subroutine for processing the token stack when id is known. Here the
        proceedure is simple - keep popping tokens onto the match stack until
        we encounter a token with a different id.

        Args:
            active : ``Token``
                The active token.

            token_stack : ``deque``
                The stack of remaining tokens

        Returns:
            The stack of tokens which match the entity.
        """
        id = active._.id
        match_stack = deque()
        match_stack.append((active, id))
        while True:
            if token_stack:
                tmp = token_stack.pop()
            else:
                break
            if tmp._.id != active._.id:
                token_stack.append(tmp)
                break
            else:
                match_stack.append((tmp, id))
        return match_stack

    def _unknown_id(self,
                    active: Token,
                    token_stack: Deque[Token]) -> Deque[Tuple[Token, str]]:
        """Subroutine for processing the token stack when id is unknown. Here
        we push tokens onto the stack as long as theret may potentially be a
        match. If there is not potential for a match, then the alias lookup
        will raise an error. Otherwise it will either return an id if a whole
        alias has been matched, or ``None`` if only part of an alias has been
        matched.

        After an error has been raised a final token is popped onto the stack -
        this is to ensure that there is at least one token to discard in
        subsequent steps.

        Args:
            active : ``Token``
                The active token.

            token_stack : ``deque``
                The stack of remaining tokens

        Returns:
            The stack of tokens which potentially match an entity.
        """
        match_stack = deque()
        while True:
            # Check if current token potentially matches something
            try:
                id = self._alias_lookup.step(active.text)
            except IndexError:
                match_stack.append((active, None))
                return match_stack
            else:
                match_stack.append((active, id))
            # Get the next token
            if token_stack:
                active = token_stack.pop()
            else:
                return match_stack

    def _annotate_tokens(self, doc: Doc) -> Doc:
        """Annotates tokens by iteratively expanding the knowledge graph and
        matching aliases.

        Args:
            doc : ``Doc``
                The document to annotate. Modified in place.
        """
        token_stack = deque(reversed(doc))
        n = len(token_stack)
        while token_stack:
            active = token_stack.pop()
            id = active._.id
            if id is None:
                logger.debug('Encountered token with unknown id')
                match_stack = self._unknown_id(active, token_stack)
            else:
                logger.debug('Encountered token with existing id "%s"', id)
                match_stack = self._existing_id(active, token_stack)
            logger.debug('Match stack: %s', match_stack)

            # Pop non-matching tokens (except the bottom-most token) back onto
            # the token stack.
            logger.debug('Discarding unmatched tokens at end of match stack')
            while len(match_stack) > 1:
                _, id = match_stack[-1]
                if id is None:
                    tmp, _ = match_stack.pop()
                    token_stack.append(tmp)
                else:
                    break
            logger.debug('Match stack: %s', match_stack)
            _, id = match_stack[-1]

            # If there's no match skip the next phase and pop a new token off
            # the stack.
            if id is None:
                continue

            # Otherwise get annotation data. Orphans and previously observed
            # entities generate themselves, otherwise lookup their "latest"
            # parent.
            logger.debug('Current stack has id "%s". Looking up story...', id)
            loc = n - len(token_stack) + len(match_stack)
            if id in self._last_seen :
                logger.debug('Id has been seen before')
                relation = ('@@REFLEXIVE@@', None)
                parent_id = id
            elif id not in self._parents:
                logger.debug('Id is not in expanded graph')
                relation = ('@@NEW@@', None)
                parent_id = id
            else:
                logger.debug('Id has been seen before')
                relation, parent_id = self._parents[id]
                logger.debug('Has parent "%s"', id)
                # Check distance cutoff. If distance is too great than relation
                # is reflexive (provided we are looking at an entity instead of
                # a literal).
                parent_loc = self._last_seen[parent_id]
                if (loc - parent_loc) > self._distance_cutoff:
                    if id not in self._alias_db:
                        continue
                    else:
                        relation = ('@@CUTOFF@@', None)
                        parent_id = id

            # If we've survived to this point then there is a valid match for
            # everything remaining in the match stack, and so we should
            # annotate.
            self._expand(id, loc)
            for token, _ in match_stack:
                token._.id = id
                token._.relation = relation
                token._.parent_id = parent_id
                if token._.source is None:
                    token._.source = 'KG'

    def _serialize_annotations(self, doc: Doc) -> Doc:
        """Serializes annotation data.

        Args:
            doc : ``Doc``
                The document to annotate. Modified in place.
        """
        annotations = []
        prev_annotation = None
        start = None
        for i, token in enumerate(doc):
            annotation = _extract_annotation(token)
            logger.debug('%s - %s', token.text, annotation)
            if annotation != prev_annotation:
                if prev_annotation is not None:
                    prev_annotation['span'] = [start, i]
                    annotations.append(prev_annotation)
                else:
                    start = i
            prev_annotation = annotation
        if prev_annotation is not None:
            prev_annotation['span'] = [start, len(doc)]
            annotations.append(prev_annotation)
        return annotations

    def annotate(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Annotates a line from the source data file.

        Source data is expected to come in JSON-lines format. Each object
        should have the following structure:

            {
                "title": "title of the article"
                "tokens": ["tokenized article text"],
                "entities": [
                    [id, start, end],
                    ...
                    [id, start, end]
                ]
                "clusters": [
                    [
                        [start, end],
                        ...
                        [start, end]
                    ],
                    ...
                    [
                        [start, end],
                        ...
                        [start, end]
                    ]
                ]
            }
        """
        self._reset()

        key = _format_wikilink(json_data['title'])

        try:
            root_id = self._wiki_db[key]
        except KeyError:
            logger.warning('Could not find entity "%s"', key)
            json_data['annotations'] = []
            return json_data
        else:
            self._add_aliases(root_id)

        doc = self._json_to_doc(json_data)
        self._propagate_ids(doc)

        self._expand(root_id, 0)
        self._annotate_tokens(doc)

        annotations = self._serialize_annotations(doc)
        json_data['annotations'] = annotations

        return json_data


def worker(q: JoinableQueue, i: int, output, print_lock: Lock, FLAGS: Tuple[Any]) -> None:
    """Retrieves files from the queue and annotates them."""
    annotator = Annotator(alias_db=SqliteDict(FLAGS.alias_db, flag='r'),
                          relation_db=SqliteDict(FLAGS.relation_db, flag='r'),
                          wiki_db=SqliteDict(FLAGS.wiki_db, flag='r'),
                          distance_cutoff=FLAGS.cutoff,
                          match_aliases=FLAGS.match_aliases,
                          unmatch_shady_nel=FLAGS.unmatch_shady_nel)
    while True:
        logger.info('Worker %i taking a task from the queue', i)
        json_data = q.get()
        if json_data is None:
            break
        annotation = annotator.annotate(json_data)
        print_lock.acquire()
        output.write(json.dumps(annotation)+'\n')
        print_lock.release()
        q.task_done()
        logger.info('Worker %i finished a task', i)

def loader(q: JoinableQueue, FLAGS: Tuple[Any]) -> None:
    with open(FLAGS.input, 'r') as f:
        for i, line in enumerate(f):
            if line == '{}\n':
                continue
            q.put(json.loads(line.strip()))
    logger.info('All %i lines have been loaded into the queue', i+1)


def main(_): # pylint: disable=missing-docstring
    logger.info('Starting queue loader')
    q = JoinableQueue()
    l = Process(target=loader, args=(q, FLAGS))
    l.start()

    logger.info('Launching worker processes')
    print_lock = Lock()
    output = open(FLAGS.output, 'w')
    processes = [Process(target=worker, args=(q, i, output, print_lock, FLAGS)) for i in range(FLAGS.j)]
    for p in processes:
        p.start()

    l.join()
    q.join()
    for _ in range(FLAGS.j):
        q.put(None)
    for p in processes:
        p.join()
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser() #pylint: disable=invalid-name
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-j', type=int, default=1,
                        help='Number of processors')
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    parser.add_argument('--relation_db', type=str, default='data/relation.db')
    parser.add_argument('--wiki_db', type=str, default='data/wiki.db')
    parser.add_argument('--cutoff', '-c', type=float, default=float('inf'),
                        help='Maximum distance between related entities')
    parser.add_argument('--match_aliases', '-m', action='store_true',
                        help='Whether or not to exact match aliases of entities '
                             'whose wikipedia links appear in the document')
    parser.add_argument('--unmatch_shady_nel', '-i', action='store_true',
                        help='Whether or not to ignore ids from NEL which do '
                             'not match ids from wikipedia')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=LEVEL)

    main(_)

