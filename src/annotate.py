# pylint: disable=redefined-builtin,redefined-outer-name
"""
Adds entity annotations to JSON-lines files.
"""
import argparse
from collections import deque
import json
import logging
from multiprocessing import Process, Queue
import queue
from typing import Any, Deque, Dict, Tuple

import spacy
from spacy.tokens import Doc, Token
from sqlitedict import SqliteDict

from src.prefix_tree import PrefixTree
from src.render import process_literal

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


# Custom SpaCy extensions
Doc.set_extension('clusters', default=[])
Token.set_extension('id', default=None)
Token.set_extension('relation', default=None)
Token.set_extension('parent_id', default=None)


def _extract_annotation(token: Token) -> Tuple[str]:
    """Extracts the annotation from a ``Token``."""
    if token._.id is not None:
        return token._.id, token._.relation, token._.parent_id
    return None


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
                 distance_cutoff: int) -> None:
        self._alias_db = alias_db
        self._relation_db = relation_db
        self._wiki_db = wiki_db
        self._distance_cutoff = distance_cutoff
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
        doc = Doc(self._nlp.vocab, words=json_data['tokens'])
        for id, start, end in json_data['entities']:
            for token in doc[start:end]:
                token._.id = id
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
        changed = False
        for cluster in doc._.clusters:
            id_set = set()
            for start, end in cluster:
                id_set.update(token._.id for token in doc[start:end] if token._.id)
            if len(id_set) == 1:
                id = id_set.pop()
                for start, end in cluster:
                    for token in doc[start:end]:
                        if token._.id != id:
                            changed = True
                            token._.id = id
        return changed

    def _expand(self, id: str, loc: int) -> None:
        """Expands the session's graph by bringing in related entities/facts.

        Arg:
            id : ``str``
                Identifier for the node to expand.
            loc : ``int``
        """
        if id not in self._last_seen:
            self._add_aliases(id)
            self._add_relations(id)
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
            return  # Object is probably a literal
        for prop, value in self._relation_db[id]:
            if value['type'] == 'wikibase-entityid':
                child_id = value['value']['id']
                try:
                    child_aliases = self._alias_db[child_id]
                except KeyError:  # A child has no name?
                    logging.warning('Encountered nameless child "%s" in '
                                    'relation table for "%s"', child_id, id)
            else:
                child_id, child_aliases = process_literal(value)

            if child_id is None:
                continue

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
                match_stack = self._unknown_id(active, token_stack)
            else:
                match_stack = self._existing_id(active, token_stack)
            # Pop non-matching tokens (except the bottom-most token) back onto
            # the token stack.
            while len(match_stack) > 1:
                _, id = match_stack[-1]
                if id is None:
                    tmp, _ = match_stack.pop()
                    token_stack.append(tmp)
                else:
                    break
            _, id = match_stack[-1]
            # If there's no match skip the next phase and pop a new token off
            # the stack.
            if id is None:
                continue
            # Otherwise get annotation data. Orphans and previously observed
            # entities generate themselves, otherwise lookup their "latest"
            # parent.
            loc = n - len(token_stack) + len(match_stack)
            if (id in self._last_seen) or not (id in self._parents):
                relation = ('@@REFLEXIVE@@', None)
                parent_id = id
            else:
                relation, parent_id = self._parents[id]
                # Check distance cutoff. If distance is too great than relation
                # is reflexive (provided we are looking at an entity instead of
                # a literal).
                parent_loc = self._last_seen[parent_id]
                if (loc - parent_loc) > self._distance_cutoff:
                    if id not in self._alias_db:
                        continue
                    else:
                        relation = ('@@REFLEXIVE@@', None)
                        parent_id = id

            # Check at least one token is a noun
            no_nouns = True
            for token, _ in match_stack:
                if token.pos_ in ('NOUN', 'PROPN'):
                    no_nouns = False
            if no_nouns:
                continue

            # If we've survived to this point then there is a valid match for
            # everything remaining in the match stack, and so we should
            # annotate.
            self._expand(id, loc)
            for token, _ in match_stack:
                token._.id = id
                token._.relation = relation
                token._.parent_id = parent_id

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
                    annotations.append([prev_annotation, [start, i]])
                else:
                    start = i
            prev_annotation = annotation
        if prev_annotation is not None:
            annotations.append([prev_annotation, [start, len(doc)]])
        return annotations

    def annotate_line(self, line: str) -> Dict[str, Any]:
        """Annotates a line from the source data file.

        Source data is expected to come in JSON-lines format. Each object
        should have the following structure:

            {
                "title": "title of the article",
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
        json_data = json.loads(line.strip())
        root_id = self._wiki_db[json_data['title']]
        doc = self._json_to_doc(json_data)
        self._propagate_ids(doc)

        running = True
        while running:
            self._reset()
            self._expand(root_id, 0)
            self._annotate_tokens(doc)
            running = self._propagate_ids(doc)

        annotations = self._serialize_annotations(doc)
        json_data['annotations'] = annotations

        return json_data


def worker(q: Queue, i: int, FLAGS: Tuple[Any]) -> None:
    """Retrieves files from the queue and annotates them."""
    annotator = Annotator(alias_db=SqliteDict(FLAGS.alias_db, flag='r'),
                          relation_db=SqliteDict(FLAGS.relation_db, flag='r'),
                          wiki_db=SqliteDict(FLAGS.wiki_db, flag='r'),
                          distance_cutoff=FLAGS.cutoff)
    while True:
        try:
            fname = q.get(block=False)
        except queue.Empty:
            logger.info('Process %i finished')
            break
        with open(fname, 'r') as f, open(fname + '.tmp', 'w') as g:
            logger.info('Annotating "%s" in process %i', fname, i)
            for line in f:
                if line == '{}\n':
                    continue
                annotation = annotator.annotate_line(line)
                g.write(json.dumps(annotation)+'\n')


def main(_): # pylint: disable=missing-docstring
    logger.info('Queueing files')
    q = Queue()
    for fname in FLAGS.inputs:
        q.put(fname)

    logger.info('Launching processes')
    processes = [Process(target=worker, args=(q, i, FLAGS)) for i in range(FLAGS.j)]
    for p in processes:
        p.start()

    logger.info('Waiting for processes to finish')
    for p in processes:
        p.join()

    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser() #pylint: disable=invalid-name
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('-j', type=int, default=1)
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    parser.add_argument('--relation_db', type=str, default='data/relation.db')
    parser.add_argument('--wiki_db', type=str, default='data/wiki.db')
    parser.add_argument('--cutoff', '-c', type=float, default=float('inf'))
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    print(FLAGS)
    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=LEVEL)

    main(_)
