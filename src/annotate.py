# pylint: disable=redefined-builtin,redefined-outer-name
"""
Adds entity annotations to JSON-lines files.
"""
from typing import Any, Deque, Dict, List, Set, Tuple 
import argparse
from collections import defaultdict, deque
import json
import logging
from multiprocessing import JoinableQueue, Lock, Process
import pickle

import spacy
from spacy.tokens import Doc, Token
from sqlitedict import SqliteDict

from src.prefix_tree import PrefixTree
from src.render import process_literal
from src.util import format_wikilink, flatten_tokens, LOG_FORMAT

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


# Constants
PRONOUNS = set(['PRP', 'PRP$', 'WP', 'WP$'])
NEL_SCORE_CUTOFF = 0.5


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
        match_aliases : ``bool``
        unmatch_shady_nel : ``bool``
        prune_clusters : ``bool``
    """
    # pylint:
        # disable=too-many-instance-attributes,too-few-public-methods,no-self-use,too-many-arguments
    def __init__(self,
                 alias_db: SqliteDict,
                 relation_db: SqliteDict,
                 wiki_db: SqliteDict,
                 distance_cutoff: int = float('inf'),
                 match_aliases: bool = False,
                 unmatch_shady_nel: bool = False,
                 prune_clusters: bool = False) -> None:

        # WikiData data structures
        self._alias_db = alias_db
        self._relation_db = relation_db
        self._wiki_db = wiki_db

        # Optional annotation arguments
        self._distance_cutoff = distance_cutoff
        self._match_aliases = match_aliases
        self._unmatch_shady_nel = unmatch_shady_nel
        self._prune_clusters = prune_clusters

        # NLP pipeline to apply to text
        self._nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Reset instance specific data structures
        self._reset()

    def _reset(self):
        """
        Resets the internal state of the ``Annotator``. Should be called before
        annotating a new document.
        """
        self._last_seen = dict()  # Tracks last time entity was seen
        self._parents = defaultdict(list)  # Tracks entities of yet to be seen entities
        self._alias_lookup = PrefixTree()  # String matching data structure

    def _add_wikilinks(self,
                       doc: Doc,
                       wikilinks: List[Any]) -> Set[str]:
        """Add wikilink data to SpaCy ``Doc``."""
        wiki_ids = set()
        for id, start, end in wikilinks:
            wiki_ids.add(id)
            if self._match_aliases:
                self._add_aliases(id)
            for token in doc[start:end]:
                token._.id = id
                token._.source = 'WIKI'
        return wiki_ids

    def _add_nel(self,
                 doc: Doc,
                 nel: List[Any],
                 wiki_ids: Set[str] = None) -> None:
        """Adds named entity linker annotations to SpaCy ``Doc``."""
        for candidate in nel:
            logger.debug('NEL Candidate %s', candidate)

            if candidate['score'] < NEL_SCORE_CUTOFF:
                logger.debug('Rejected - Low Score')
                continue

            start = candidate['start']
            end = candidate['end']

            already_linked = False
            for token in doc[start:end]:
                if token._.id is not None:
                    already_linked = True
            if already_linked:
                logger.debug('Rejected - Already Linked')
                continue

            key = format_wikilink(candidate['label'])
            logger.debug('Key - %s', key)
            try:
                id = self._wiki_db[key]
            except KeyError:
                logger.debug('Rejected - Key not found')
                continue

            is_not_shady = key in wiki_ids if self._unmatch_shady_nel else True
            if key in self._wiki_db and is_not_shady:
                for token in doc[start:end]:
                    logger.debug('%s - %s', token, token._.id)
                    token._.id = id
                    token._.source = 'NEL'

    @staticmethod
    def _detect_cluster_ids(doc: Doc,
                            cluster: List[List[int]]) -> Set[str]:
        """Detects the entity ids that appear in a cluster."""
        cluster_ids = set()
        for start, end in cluster:
            mention_ids = [token._.id for token in doc[start:end+1] if token._.id]
            cluster_ids.update(mention_ids)
        return cluster_ids

    @staticmethod
    def _prune_cluster(doc: Doc,
                       cluster: List[List[int]],
                       alias_lookup: Set[str]) -> List[List[int]]:
        logger.debug('Pruning cluster')

        # Prune step
        new_cluster = []
        for start, end in cluster:
            mention = doc[start:end+1]

            # Check for pronoun
            if start == end:
                if mention[0].tag_ in PRONOUNS:
                    new_cluster.append([start, end])
                    logger.debug('Found pronoun: %s', mention)

            # Check for alias match
            mention_tokens = tuple(x.text for x in mention)
            if mention_tokens in alias_lookup:
                new_cluster.append([start, end])
                logger.debug('Exact alias match: %s', mention)
            else:
                logger.debug('No match: %s', mention)

        return new_cluster

    def _propagate_ids(self, doc: Doc,
                       clusters: List[List[int]],
                       wiki_ids: Set[str] = None) -> None:
        """Propagates id's in coreference clusters.

        Args:
            doc: ``Doc``
                Document to propagate ids in. Modified in place.
        """
        if self._prune_clusters:
            # Create an exact match tree for aliases
            for id in wiki_ids:
                alias_lookup = set()
                aliases = self._alias_db[id]
                for alias in aliases:
                    alias_tokens = tuple(x.text for x in self._nlp.tokenizer(alias))
                    alias_lookup.add(alias_tokens)

        for i, cluster in enumerate(clusters):

            # (Optional) Prune mentions / tokens from cluster which do not
            # correspond to a pronoun or entity alias
            if self._prune_clusters:
                logger.debug('Pruning cluster %i', i)
                logger.debug('Initial Spans: %s', cluster)
                cluster = self._prune_cluster(doc, cluster, alias_lookup)
                logger.debug('Surviving spans: %s', cluster)

            # Extract set of unique entity ids of mentions in the cluster
            cluster_ids = self._detect_cluster_ids(doc, cluster)
            logger.debug('Detected following ids for cluster %i: %s', i, cluster_ids)
            if len(cluster_ids) != 1:
                logger.debug('Number of ids is not equal to 1, skipping propagation')
                continue
            id = cluster_ids.pop()

            # Propagate the id
            logger.debug('Propagating id "%s" through cluster %i', id, i)
            for start, end in cluster:
                for token in doc[start:end+1]:
                    if token._.id != id:
                        if token._.id is not None:
                            raise RuntimeError('Tried to overwrite existing token id')
                        token._.id = id
                        token._.source = 'COREF'

    def _json_to_doc(self, json_data: Dict[str, Any]) -> Doc:
        """Converts a JSON object to a SpaCy ``Doc``

        Args:
            json_data : ``dict``
                Parsed JSON data from an annotation file.

        Returns:
            A SpaCy ``Doc``.
        """
        # Create SpaCy doc from tokens
        tokens = flatten_tokens(json_data['tokens'])
        doc = Doc(self._nlp.vocab, words=tokens)

        # Run NLP pipeline to get parse tree and POS tags
        for _, pipe in self._nlp.pipeline:
            doc = pipe(doc)

        # Add wikilink data
        wiki_ids = self._add_wikilinks(doc, json_data['entities'])

        # Add nel data
        self._add_nel(doc, json_data['nel'], wiki_ids)

        # Propagate links through coref clusters
        self._propagate_ids(doc, json_data['clusters'], wiki_ids)

        return doc

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
            else:
                child_id, child_aliases = process_literal(value)

            if child_id is None:
                continue

            logger.debug('Adding child id "%s" to graph w/ parent "%s"',
                         child_id, id)
            self._parents[child_id].append((prop, id))
            for child_alias in child_aliases:
                child_alias_tokens = [x.text for x in self._nlp.tokenizer(child_alias)]
                if child_alias_tokens not in self._alias_lookup:
                    self._alias_lookup.add(child_alias_tokens, child_id)

    def _expand(self, id: str, loc: int) -> None:
        """Expands the session's graph by bringing in related entities/facts.

        Arg:
            id : ``str``
                Identifier for the node to expand.
            loc : ``int``
        """
        logger.debug('Expanding id "%s" at location %i', id, loc)
        if id not in self._last_seen:
            self._add_relations(id)
        else:
            logger.debug('Node has already been expanded')
        self._last_seen[id] = loc

    @staticmethod
    def _existing_id(active: Token,
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
            if id in self._last_seen:
                logger.debug('Id has been seen before')
                relation = ['@@REFLEXIVE@@']
                parent_id = [id]
            elif id in self._parents:
                parents = self._parents[id]
                logger.debug('Has parents "%s"', parents)
                # Check distance cutoff. If distance is too great than relation
                # is reflexive (provided we are looking at an entity instead of
                # a literal).
                parent_locs = [self._last_seen[parent_id] for _, parent_id in parents]
                parents = [x for x, y in zip(parents, parent_locs) if y < self._distance_cutoff]
                if parents:
                    relation, parent_id = zip(*parents)
                    relation = list(relation)
                    parent_id = list(parent_id)
            else:
                logger.debug('Id is not in expanded graph')
                relation = ['@@NEW@@']
                parent_id = [id]

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
        key = format_wikilink(json_data['title'])

        try:
            root_id = self._wiki_db[key]
        except KeyError:
            logger.warning('Could not find entity "%s"', key)
            json_data['annotations'] = []
            return json_data
        else:
            self._add_aliases(root_id)

        doc = self._json_to_doc(json_data)

        self._expand(root_id, 0)
        self._annotate_tokens(doc)

        annotations = self._serialize_annotations(doc)
        json_data['annotations'] = annotations

        return json_data


def worker(q: JoinableQueue, i: int, output, print_lock: Lock, FLAGS: Tuple[Any]) -> None:
    """Retrieves files from the queue and annotates them."""
    if FLAGS.in_memory:
        with open(FLAGS.alias_db, 'rb') as f:
            alias_db = pickle.load(f)
        with open(FLAGS.relation_db, 'rb') as f:
            relation_db = pickle.load(f)
        with open(FLAGS.wiki_db, 'rb') as f:
            wiki_db = pickle.load(f)
    else:
        alias_db = SqliteDict(FLAGS.alias_db, flag='r')
        relation_db = SqliteDict(FLAGS.relation_db, flag='r')
        wiki_db = SqliteDict(FLAGS.wiki_db, flag='r')

    annotator = Annotator(alias_db,
                          relation_db,
                          wiki_db,
                          distance_cutoff=FLAGS.cutoff,
                          match_aliases=FLAGS.match_aliases,
                          unmatch_shady_nel=FLAGS.unmatch_shady_nel,
                          prune_clusters=FLAGS.prune_clusters)
    while True:
        logger.debug('Worker %i taking a task from the queue', i)
        json_data = q.get()
        if json_data is None:
            break
        annotation = annotator.annotate(json_data)
        print_lock.acquire()
        output.write(json.dumps(annotation)+'\n')
        print_lock.release()
        q.task_done()
        logger.debug('Worker %i finished a task', i)


def loader(q: JoinableQueue, FLAGS: Tuple[Any]) -> None:
    i = 0
    with open(FLAGS.input, 'r') as f:
        for i, line in enumerate(f):
            if line == '{}\n':
                continue
            q.put(json.loads(line.strip()))
    logger.info('All %i lines have been loaded into the queue', i+1)


def main(_): # pylint: disable=missing-docstring
    if FLAGS.prune_clusters:
        logger.warning('Cluster pruning is active. Beware the doc._.clusters is '
                       'not updated to only hold pruned mentions - this may '
                       'cause unexpected behavior if ``Annotator._propagate_ids()`` '
                       'is called multiple times.')
    logger.info('Starting queue loader')
    q = JoinableQueue(maxsize=256)
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
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--cutoff', '-c', type=float, default=float('inf'),
                        help='Maximum distance between related entities')
    parser.add_argument('--match_aliases', '-m', action='store_true',
                        help='Whether or not to exact match aliases of entities '
                             'whose wikipedia links appear in the document')
    parser.add_argument('--unmatch_shady_nel', '-i', action='store_true',
                        help='Whether or not to ignore ids from NEL which do '
                             'not match ids from wikipedia')
    parser.add_argument('--prune_clusters', '-p', action='store_true',
                        help='Whether or not to prune mentions from clusters '
                             'that are not aliases or pronouns')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=LEVEL)

    main(_)

