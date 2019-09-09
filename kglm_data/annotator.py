import re
import logging
from typing import Any, Deque, Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from time import time
from importlib import import_module

import spacy
from sqlitedict import SqliteDict
from spacy.tokens import Doc, Token
from spacy.util import get_lang_class, load_model_from_path

from kglm_data.prefix_tree import PrefixTree
from kglm_data.render import process_literal
from kglm_data.util import format_wikilink, flatten_tokens, merge_numbers_to_single_token


PRONOUNS_PER_LANGUAGE = {'sv': {'HP', 'HS', 'PN', 'PS'},
                         'en': {'PRP', 'PRP$', 'WP', 'WP$'}}
NEL_SCORE_CUTOFF = 0.5
RE_ENTITY = re.compile(r'Q\d+')

# Custom SpaCy extensions
Doc.set_extension('clusters', default=[])
Token.set_extension('source', default=None)
Token.set_extension('id', default=None)
Token.set_extension('relation', default=None)
Token.set_extension('parent_id', default=None)


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
        unmatch : ``bool``
        prune_clusters : ``bool``
    """
    # pylint: disable=too-many-instance-attributes,too-few-public-methods,no-self-use,too-many-arguments
    def __init__(self,
                 alias_db: SqliteDict,
                 relation_db: SqliteDict,
                 wiki_db: SqliteDict,
                 distance_cutoff: int = float('inf'),
                 match_aliases: bool = False,
                 unmatch: bool = False,
                 prune_clusters: bool = False,
                 language: str = "en",
                 merge_entities: bool = False,
                 spacy_model_path: str = False,
                 logger: logging.Logger = None) -> None:

        self.STOP_WORDS = getattr(import_module(f"spacy.lang.{language}.stop_words"), "STOP_WORDS")
        self.PRONOUNS = PRONOUNS_PER_LANGUAGE.get(language, set())

        self.logger = logger or logging.getLogger(__name__)  # pylint: disable=invalid-name

        # WikiData data structures
        self._alias_db = alias_db
        self._relation_db = relation_db
        self._wiki_db = wiki_db

        # Optional annotation arguments
        self._distance_cutoff = distance_cutoff
        self._match_aliases = match_aliases
        self._unmatch = unmatch
        self._prune_clusters = prune_clusters
        self._merge_entities = merge_entities

        # NLP pipeline to apply to text
        self._nlp = self._create_nlp_pipeline(language, spacy_model_path)
        self._nlp.add_pipe(merge_numbers_to_single_token, after="tagger")

        if merge_entities:
            merge_ents = self._nlp.create_pipe("merge_entities")
            self._nlp.add_pipe(merge_ents)

        # Reset instance specific data structures
        self._reset()

    @staticmethod
    def _create_nlp_pipeline(language: str, model_path: str = None):
        if not model_path:
            return spacy.load('en', disable=['parser', 'ner'])
        else:
            return load_model_from_path(model_path, disable=['parser'])

    def _reset(self):
        """
        Resets the internal state of the ``Annotator``. Should be called before
        annotating a new document.
        """
        self._last_seen = dict()  # Tracks last time entity was seen
        self._parents = defaultdict(set)  # Tracks parents of yet to be seen entities
        self._alias_lookup = PrefixTree()  # String matching data structure

    def _add_wikilinks(self,
                       doc: Doc,
                       wikilinks: List[Any]) -> Set[str]:
        """Add wikilink data to SpaCy ``Doc``."""
        wiki_ids = set()
        for wiki_id, start, end in wikilinks:
            wiki_ids.add(wiki_id)
            if self._match_aliases:
                self._add_aliases(wiki_id)
            for token in doc[start:end]:
                token._.id = wiki_id
                token._.source = 'WIKI'
        return wiki_ids

    def _add_nel(self,
                 doc: Doc,
                 nel: List[Any],
                 wiki_ids: Set[str] = None) -> None:
        """Adds named entity linker annotations to SpaCy ``Doc``."""
        for candidate in nel:
            self.logger.debug('NEL Candidate %s', candidate)

            if candidate['score'] < NEL_SCORE_CUTOFF:
                self.logger.debug('Rejected - Low Score')
                continue

            start = candidate['start']
            end = candidate['end']

            already_linked = False
            for token in doc[start:end]:
                if token._.id is not None:
                    already_linked = True
            if already_linked:
                self.logger.debug('Rejected - Already Linked')
                continue

            key = format_wikilink(candidate['label'])
            logging.debug('Key - %s', key)
            try:
                wiki_id = self._wiki_db[key]
            except KeyError:
                logging.debug('Rejected - Key not found')
                continue
            matches = wiki_id in wiki_ids if self._unmatch else True
            if matches:
                for token in doc[start:end]:
                    self.logger.debug('%s - %s', token, wiki_id)
                    token._.id = wiki_id
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

    def _prune_cluster(self,
                       doc: Doc,
                       cluster: List[List[int]],
                       alias_token_set: Set[str]) -> List[List[int]]:
        self.logger.debug('Pruning cluster')

        # Prune step
        new_cluster = []
        for start, end in cluster:
            mention = doc[start:end+1]

            # Check for pronoun
            if start == end:
                tag = mention[0].tag_
                if set(tag.split("|")).intersection(self.PRONOUNS):
                    new_cluster.append([start, end])
                    self.logger.debug('Found pronoun: %s', mention)
                    continue

            # Check for alias match
            mention_tokens = tuple(t.text for t in mention)
            mention_types = tuple(x.tag_ for x in mention)
            keep = True
            for token, type in zip(mention_tokens, mention_types):
                not_in_alias = token not in alias_token_set
                not_det = type != 'DT'
                if not_in_alias and not_det:
                    self.logger.debug('Encountered invalid token: "%s" - rejecting', token)
                    keep = False
                    break
            if keep:
                new_cluster.append([start, end])

        return new_cluster

    def _propagate_ids(self,
                       doc: Doc,
                       clusters: List[List[List[int]]]) -> None:
        """Propagates id's in coreference clusters.

        Args:
            doc: ``Doc``
                Document to propagate ids in. Modified in place.
        """

        for i, cluster in enumerate(clusters):

            # (Optional) Prune mentions / tokens from cluster which do not
            # correspond to a pronoun or entity alias
            if self._prune_clusters:

                # Extract set of unique entity ids of mentions in the cluster
                pre_pruned_cluster_ids = self._detect_cluster_ids(doc, cluster)
                self.logger.debug('Detected following prepruned ids for cluster %i: %s', i, pre_pruned_cluster_ids)
                if len(pre_pruned_cluster_ids) != 1:
                    self.logger.debug('Number of prepruned ids is not equal to 1, skipping propagation')
                    continue
                pre_pruned_cluster_id = pre_pruned_cluster_ids.pop()

                # Create an alias token set
                alias_token_set = set()
                aliases = self._alias_db[pre_pruned_cluster_id]
                alias_docs = self._nlp.tokenizer.pipe(aliases)
                for alias_doc in alias_docs:
                    if self._bad_alias(alias_doc):
                        continue
                    alias_tokens = tuple(x.text for x in alias_doc)
                    alias_token_set.update(alias_tokens)
                    alias_token_set.update(self.capitalize(alias_tokens))
                self.logger.debug('Alias token set: %s', alias_token_set)

                self.logger.debug('Pruning cluster %i', i)
                self.logger.debug('Initial Spans: %s', cluster)
                cluster = self._prune_cluster(doc, cluster, alias_token_set)
                self.logger.debug('Surviving spans: %s', cluster)

            # Extract set of unique entity ids of mentions in the cluster
            cluster_ids = self._detect_cluster_ids(doc, cluster)
            self.logger.debug('Detected following ids for cluster %i: %s', i, cluster_ids)
            if len(cluster_ids) != 1:
                self.logger.warning(f'Number of ids is not equal to 1 for cluster {i}, skipping propagation')
                continue
            wiki_id = cluster_ids.pop()

            # Propagate the id
            self.logger.debug('Propagating id "%s" through cluster %i', wiki_id, i)
            for start, end in cluster:
                for token in doc[start:end+1]:
                    if token._.id != wiki_id:
                        if token._.id is not None:
                            raise RuntimeError('Tried to overwrite existing token id')
                        token._.id = wiki_id
                        token._.source = 'COREF'

    def _json_to_doc(self, json_data: Dict[str, Any], root_id: Optional[str] = None) -> Doc:
        """Converts a JSON object to a SpaCy ``Doc``

        Args:
            json_data : ``dict``
                Parsed JSON data from an annotation file.

        Returns:
            A SpaCy ``Doc``.
        """
        # Create SpaCy doc from tokens
        tokens = flatten_tokens(json_data['tokens'])
        self.logger.info(f"Before flattening '{json_data['title']}': "
                         f"{len(json_data['tokens'])} tokens, after: {len(tokens)}")
        doc = Doc(self._nlp.vocab, words=tokens)

        # Run NLP pipeline to get parse tree and POS tags
        for _, pipe in self._nlp.pipeline:
            doc = pipe(doc)

        # Add wikilink data
        wiki_ids = self._add_wikilinks(doc, json_data['entities'])
        if root_id is not None:
            wiki_ids.add(root_id)  # STUPID HACK: Prevents unmatching article subject

        # Add nel data
        try:
            self._add_nel(doc, json_data['nel'], wiki_ids)
        except KeyError:
            self.logger.warning(f"No NEL data for {json_data['title']}")

        # Propagate links through coref clusters
        try:
            self._propagate_ids(doc, json_data['clusters'])
        except KeyError:
            self.logger.warning(f"No COREF data for {json_data['title']}")

        return doc

    def _add_aliases(self, wiki_id: str) -> None:
        """
        [?]
        Adds aliases for the given node to the alias lookup.

        This seems to only run for the root node, not any related nodes

        Args:
            wiki_id: ``str``
                Identifier for the node.
        """
        t0 = time()
        try:
            aliases = self._alias_db[wiki_id]
        except KeyError:  # Object is probably a literal
            self.logger.debug('No aliases found for node "%s"', wiki_id)
        else:
            alias_docs = self._nlp.tokenizer.pipe(aliases)
            for alias_doc in alias_docs:
                if self._bad_alias(alias_doc):
                    continue
                alias_tokens = [x.text for x in alias_doc]
                if alias_tokens not in self._alias_lookup:
                    self._alias_lookup.add(alias_tokens, wiki_id)
                    self._alias_lookup.add(self.capitalize(alias_tokens), wiki_id)
        self.logger.debug(f"Added aliases for {wiki_id} in {time()-t0} seconds")

    def _add_relations(self, wiki_id: str) -> None:
        """
        [SLOW]
        Adds relations for the given node to the relation (parent) lookup.

        Args:
            wiki_id: ``str``
                Identifier for the node.
        """
        t0 = time()
        count_relations = 0
        count_aliases = 0
        if wiki_id not in self._relation_db:
            self.logger.debug('Node is not in the relation db!')
            return  # Object is probably a literal
        for prop, value in self._relation_db[wiki_id]:
            count_relations += 1

            # Fetch aliases of the related entity
            if value['type'] == 'wikibase-entityid' and value.get('entity-type') != 'property':
                child_id = value['value']['id']
                try:
                    child_aliases = self._alias_db[child_id]
                except:
                    continue
            else:
                child_id, child_aliases = process_literal(value, self._alias_db)
                if child_id is None:
                    continue

            count_aliases += len(child_aliases)
            child_alias_docs = self._nlp.tokenizer.pipe(child_aliases)

            if prop == 'P395':
                self.logger.debug('Handling Commons relation')
                for child_alias_doc in child_alias_docs:
                    if self._bad_alias(child_alias_doc):
                        continue
                    child_alias_tokens = [x.text for x in child_alias_doc]
                    if child_alias_tokens not in self._alias_lookup:
                        self._alias_lookup.add(child_alias_tokens, wiki_id)
                        self._alias_lookup.add(self.capitalize(child_alias_tokens), wiki_id)
                continue

            self.logger.debug('Adding child id "%s" to graph w/ parent "%s"', child_id, wiki_id)
            self._parents[child_id].add((prop, wiki_id))
            for child_alias_doc in child_alias_docs:
                child_alias_tokens = [x.text for x in child_alias_doc]
                self.logger.debug('Child alias tokens %s', child_alias_tokens)
                if self._bad_alias(child_alias_doc):
                    self.logger.debug('Rejected')
                    continue
                if child_alias_tokens not in self._alias_lookup:
                    self._alias_lookup.add(child_alias_tokens, child_id)
                    self._alias_lookup.add(self.capitalize(child_alias_tokens), child_id)
        self.logger.debug(f"Added {count_relations} relations with {count_aliases} aliases in {time() - t0} seconds")

    def _expand(self, wiki_id: str, loc: int) -> None:
        """
        [SLOW]
        Expands the session's graph by bringing in related entities/facts.

        Arg:
            id : ``str``
                Identifier for the node to expand.
            loc : ``int``
        """
        t0 = time()
        self.logger.debug('Expanding id "%s" at location %i', wiki_id, loc)
        if wiki_id not in self._last_seen:
            self._add_relations(wiki_id)
        else:
            self.logger.debug('Node has already been expanded')
        self._last_seen[wiki_id] = loc
        self.logger.debug(f'Expanded {wiki_id} at location {loc} in {time() - t0} seconds')

    @staticmethod
    def _existing_id(active: Token,
                     token_stack: Deque[Token]) -> Deque[Tuple[Token, str]]:
        """
        [Optimised]
        Subroutine for processing the token stack when id is known. Here the
        procedure is simple - keep popping tokens onto the match stack until
        we encounter a token with a different id.

        Args:
            active : ``Token``
                The active token.

            token_stack : ``deque``
                The stack of remaining tokens

        Returns:
            The stack of tokens which match the entity.
        """
        wiki_id = active._.id
        match_stack = deque()
        match_stack.append((active, wiki_id))
        while True:
            if token_stack:
                tmp = token_stack.pop()
            else:
                break
            if tmp._.id != active._.id:
                token_stack.append(tmp)
                break
            else:
                match_stack.append((tmp, wiki_id))
        return match_stack

    def _unknown_id(self,
                    active: Token,
                    token_stack: Deque[Token]) -> Deque[Tuple[Token, str]]:
        """
        [Optimised]
        Subroutine for processing the token stack when id is unknown. Here
        we push tokens onto the stack as long as there may potentially be a
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
                wiki_id = self._alias_lookup.step(active.text)
            except IndexError:
                match_stack.append((active, None))
                return match_stack
            else:
                match_stack.append((active, wiki_id))
            # Get the next token
            if token_stack:
                active = token_stack.pop()
            else:
                return match_stack

    def _annotate_tokens(self, doc: Doc) -> None:
        """Annotates tokens by iteratively expanding the knowledge graph and
        matching aliases.

        Args:
            doc : ``Doc``
                The document to annotate. Modified in place.
        """
        self.logger.info("Annotating tokens")
        token_stack = deque(reversed(doc))
        n = len(token_stack)
        while token_stack:
            active = token_stack.pop()
            wiki_id = active._.id
            if wiki_id is None:
                self.logger.debug('Encountered token with unknown id')
                match_stack = self._unknown_id(active, token_stack)
            else:
                self.logger.debug('Encountered token with existing id "%s"', wiki_id)
                match_stack = self._existing_id(active, token_stack)
            self.logger.debug('Match stack: %s', match_stack)

            # Append non-matching tokens (except the bottom-most token) back onto
            # the token stack.
            self.logger.debug('Discarding unmatched tokens at end of match stack')
            while len(match_stack) > 1:
                _, wiki_id = match_stack[-1]
                if wiki_id is None:
                    tmp, _ = match_stack.pop()
                    token_stack.append(tmp)
                else:
                    break
            self.logger.debug('Match stack: %s', match_stack)
            _, wiki_id = match_stack[-1]

            # If there's no match skip the next phase and pop a new token off
            # the stack.
            if wiki_id is None:
                continue

            # Otherwise get annotation data. Orphans and previously observed
            # entities generate themselves, otherwise lookup their "latest"
            # parent.
            self.logger.debug('Current stack has id "%s". Looking up story...', wiki_id)
            loc = n - len(token_stack) + len(match_stack)
            relation = []
            parent_id = []
            if wiki_id in self._last_seen:
                self.logger.debug('Id has been seen before at location: %i',
                             self._last_seen[wiki_id])
                if (loc - self._last_seen[wiki_id]) > self._distance_cutoff:
                    self.logger.debug('...but past cutoff')
                else:
                    relation.append('@@REFLEXIVE@@')
                    parent_id.append(wiki_id)
            if wiki_id in self._parents:
                parents = self._parents[wiki_id]
                self.logger.debug('Has parents "%s"', parents)
                # Check distance cutoff. If distance is too great then relation
                # is reflexive (provided we are looking at an entity instead of
                # a literal).
                parent_locs = [(loc - self._last_seen[parent_id]) for _, parent_id in parents]
                self.logger.debug('With locs %s', parent_locs)
                parents = [(x, y) for x, y in zip(parents, parent_locs) if y < self._distance_cutoff]
                parents.sort(key=lambda x: x[1])
                parents = [x for x, y in parents][:10]
                self.logger.debug('Has surviving parents "%s"', parents)
                if parents:
                    _relation, _parent_id = zip(*parents)
                    relation.extend(list(_relation))
                    parent_id.extend(list(_parent_id))
            if len(relation) == 0:
                self.logger.debug('No surviving story')
                if RE_ENTITY.match(wiki_id) and any(token._.source is not None for token, _ in match_stack):
                    relation = ['@@NEW@@']
                    parent_id = [wiki_id]
                else:
                    self.logger.debug('Not an entity, rejecting')
                    for token, _ in match_stack:
                        token._.id = None
                        token._.relation = None
                        token._.parent_id = None
                        token._.source = None
                    continue

            # If we've survived to this point then there is a valid match for
            # everything remaining in the match stack, and so we should
            # annotate.
            self.logger.debug(f"Annotating {[x[0] for x in match_stack]} ({wiki_id}) with {[x for x in zip(relation, parent_id)]}")
            for token, _ in match_stack:
                token._.id = wiki_id
                token._.relation = relation
                token._.parent_id = parent_id
                if token._.source is None:
                    token._.source = 'KG'
            # Lastly expand the current node, but only if it really corresponds
            # to an entity
            if wiki_id in self._alias_db:
                self._expand(wiki_id, loc)

    def _serialize_annotations(self, doc: Doc) -> List[Dict[str, Any]]:
        """Serializes annotation data.

        Annotations are added on a token-by-token bases earlier. Here we expand
        the annotations to cover spans of tokens instead.

        Args:
            doc : ``Doc``
                The document to annotate. Modified in place.
        """
        annotations = []
        prev_annotation = None
        start = None
        for i, token in enumerate(doc):
            annotation = self._extract_annotation(token)
            self.logger.debug('%s - %s', token.text, annotation)
            if annotation != prev_annotation:
                if prev_annotation is not None:
                    prev_annotation['span'] = [start, i]
                    annotations.append(prev_annotation)
                start = i
            prev_annotation = annotation
        if prev_annotation is not None:
            prev_annotation['span'] = [start, len(doc)]
            annotations.append(prev_annotation)
        return annotations

    def _bad_alias(self, tokens: Tuple[Token]) -> bool:
        if all(x.tag_ in self.PRONOUNS for x in tokens):
            return True
        if all(x.pos_ == 'PUNCT' for x in tokens):
            return True
        if all(x.text in self.STOP_WORDS for x in tokens):
            return True
        if all(x.text.lower() in self.STOP_WORDS for x in tokens):
            return True
        return False

    @staticmethod
    def _extract_annotation(token: Token) -> Optional[Dict[str, Any]]:
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

    @staticmethod
    def capitalize(tokens: List[str]) -> Tuple[str]:
        if len(tokens) > 1:
            return (tokens[0].capitalize(), *tokens[1:])
        else:
            return tokens[0].capitalize(),

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
                ],
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
                ],
                "nel": [
                    {'tokens': str,
                      'end': int,
                      'start': int,
                      'jointScoreMap': {str: int},
                      'contextScoreMap': {str: int},
                      'priorScoreMap': {str: int},
                      'label': 'str',
                      'score': int
                      },
                      ...
                      {
                      ...
                      }
                ]
            }
        """
        self._reset()
        key = format_wikilink(json_data['title'])

        try:
            root_id = self._wiki_db[key]
        except KeyError:
            self.logger.warning('Could not find entity "%s"', key)
            json_data['annotations'] = []
            root_id = None
        else:
            self._add_aliases(root_id)
            self.logger.debug('Alias lookup: %s', self._alias_lookup._root)

        doc = self._json_to_doc(json_data, root_id)

        self._annotate_tokens(doc)

        annotations = self._serialize_annotations(doc)
        json_data['annotations'] = annotations

        return json_data


class ArticleAnnotator(Annotator):
    """
    Annotator that processes news articles (where there is no root id).
    """

    def annotate(self, json_data: Dict[str, Any]):
        self._reset()

        doc = self._json_to_doc(json_data)

        self._annotate_tokens(doc)

        annotations = self._serialize_annotations(doc)
        json_data['annotations'] = annotations

        return json_data
