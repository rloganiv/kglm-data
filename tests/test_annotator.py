from collections import deque
import json
import logging
import os
import unittest
import sys

from spacy.tokens import Doc, Token

from kglm_data.annotator import Annotator
from kglm_data.util import flatten_tokens

logger = logging.getLogger(__name__)


# Fixtures
ALIAS_DB = {
    'Q0': ['Robert Plant'],
    'Q1': ['Robert Lockwood Logan IV', 'Robby'],
    'Q2': ['Robert Lockwood'],
    'Q3': ['Led Zeppelin']
}

WIKI_DB = {
    'Robert_Plant': 'Q0',
    'Robert_Lockwood_Logan_IV': 'Q1',
    'Robert_Lockwood': 'Q2',
    'Led_Zeppelin': 'Q3',
    'Singing': 'Q4'
}

RELATION_DB = {
    'Q0': [('influenced_by',
            {'type': 'wikibase-entityid', 'value':{'id':'Q2'}}),
           ('plays_in',
            {'type': 'wikibase-entityid', 'value':{'id':'Q3'}})
          ],
    'Q1': [('listens_to',
            {'type': 'wikibase-entityid', 'value':{'id':'Q0'}})],
    'Q2': []
}

CUTOFF = 500
LANGUAGE = 'en'
SPACY_PATH = False


class TestStandardAnnotator(unittest.TestCase):

    def setUp(self):
        self.alias_db = ALIAS_DB
        self.relation_db = RELATION_DB
        self.wiki_db = WIKI_DB
        self.distance_cutoff = CUTOFF
        self.match_aliases = True
        self.unmatch = False
        self.prune_clusters = True
        self.language = LANGUAGE
        self.merge_entities = False
        self.spacy_model_path = SPACY_PATH
        self.annotator = Annotator(alias_db=self.alias_db,
                                   relation_db=self.relation_db,
                                   wiki_db=self.wiki_db,
                                   distance_cutoff=self.distance_cutoff,
                                   match_aliases=self.match_aliases,
                                   unmatch=self.unmatch,
                                   prune_clusters=self.prune_clusters,
                                   language=self.language,
                                   merge_entities=self.merge_entities,
                                   spacy_model_path=self.spacy_model_path)

        with open('/tests/fixtures/test.jsonl', 'r') as f:
            test_line = f.readline()
        self.json_data = json.loads(test_line.strip())
        tokens = flatten_tokens(self.json_data['tokens'])
        doc = Doc(self.annotator._nlp.vocab, words=tokens)
        for _, pipe in self.annotator._nlp.pipeline:
            doc = pipe(doc)
        self.doc = doc

        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    def test_reset(self):
        """
        Ensures that ``Annotator.reset()`` erases the current state.
        """
        # Add some data from the KG to the state.
        self.annotator._add_aliases('Q1')
        self.annotator._add_relations('Q1')
        # Reset and check relevant attributes are empty
        self.annotator._reset()
        self.assertEqual(self.annotator._last_seen, {})
        self.assertEqual(self.annotator._parents, {})
        self.assertEqual(len(self.annotator._alias_lookup._root), 0)

    def test_add_wikilinks(self):
        """
        Ensures that ids from wikilinks are added to corresponding tokens.
        """
        self.annotator._add_wikilinks(self.doc, self.json_data['entities'])
        self.assertEqual(self.doc[0]._.id, 'Q0')
        self.assertEqual(self.doc[1]._.id, 'Q0')

    def test_add_nel(self):
        """
        Ensures that ids from nel are added to corresponding tokens.
        """
        self.annotator._add_nel(self.doc, self.json_data['nel'])
        # "Robert Plant" should be linked, since the wikilinks have not been
        # added
        self.assertEqual(self.doc[0]._.id, 'Q0')
        self.assertEqual(self.doc[1]._.id, 'Q0')
        # "sings" should not get linked, since its score is too low
        self.assertNotEqual(self.doc[2]._.id, 'Q4')
        # "Led Zeppelin" should be linked, no matter what
        self.assertEqual(self.doc[4]._.id, 'Q3')
        self.assertEqual(self.doc[5]._.id, 'Q3')

    def test_add_nel_avoids_overwriting_wikilinks(self):
        """
        Ensures that wikilink annotations are not overwritten by nel
        annotations.
        """
        self.annotator._add_wikilinks(self.doc, self.json_data['entities'])
        self.annotator._add_nel(self.doc, self.json_data['nel'])
        # "Robert Plant"'s source should be WIKI
        self.assertEqual(self.doc[0]._.source, 'WIKI')
        self.assertEqual(self.doc[1]._.source, 'WIKI')
        # "Led Zeppelin"'s source should be NEL
        self.assertEqual(self.doc[4]._.source, 'NEL')
        self.assertEqual(self.doc[5]._.source, 'NEL')

    def test_detect_cluster_ids(self):
        """
        Ensures that cluster ids are properly detected.
        """
        self.annotator._add_wikilinks(self.doc, self.json_data['entities'])
        self.annotator._add_nel(self.doc, self.json_data['nel'])
        # Pretend cluster contains "Robert Plant", "Led Zeppelin" and "He"
        cluster = [[0,1], [4,5], [7,7]]
        cluster_ids = self.annotator._detect_cluster_ids(self.doc, cluster)
        self.assertSetEqual(cluster_ids, {'Q0', 'Q3'})

    def test_prune_cluster(self):
        # Assume "Led Zeppelin" is not a viable alias, then it should be
        # pruned.
        cluster = [[0,1], [4,5], [7,7]]
        alias_token_set = {'Robert', 'Plant'}
        cluster = self.annotator._prune_cluster(self.doc, cluster, alias_token_set)
        self.assertListEqual(cluster, [[0, 1], [7, 7]])

    def test_propagate_ids(self):
        """
        Tests that cluster ids are properly propagated.
        """
        # Propagate through a 'good' cluster
        self.annotator._add_wikilinks(self.doc, self.json_data['entities'])
        self.annotator._add_nel(self.doc, self.json_data['nel'])
        self.annotator._propagate_ids(self.doc, self.json_data['clusters'])
        self.assertEqual(self.doc[7]._.id, 'Q0')

    def test_dont_propagate_ids(self):
        # Check that propagation doesn't happen for a bad cluster.
        bad_clusters = [[[0,1], [4,5], [7,7]]]
        self.annotator._add_wikilinks(self.doc, self.json_data['entities'])
        self.annotator._add_nel(self.doc, self.json_data['nel'])
        self.annotator._propagate_ids(self.doc, bad_clusters)
        self.assertIsNone(self.doc[7]._.id)

    def test_json_to_doc(self):
        """Tests that the whole pipeline works (without any flags enabled)."""
        doc = self.annotator._json_to_doc(self.json_data, root_id='Q0')
        self.assertEqual(doc[0]._.id, 'Q0')
        self.assertEqual(doc[0]._.source, 'WIKI')
        self.assertEqual(doc[4]._.id, 'Q3')
        self.assertEqual(doc[4]._.source, 'NEL')
        self.assertEqual(doc[7]._.id, 'Q0')
        self.assertEqual(doc[7]._.source, 'COREF')

    def test_add_aliases(self):
        """
        Tests that aliases are properly added to ``Annotator._alias_lookup``
        """
        self.annotator._add_aliases('Q1')
        good_alias_0 = [x.text for x in self.annotator._nlp.tokenizer('Robert Lockwood Logan IV')]
        self.assertIn(good_alias_0, self.annotator._alias_lookup)
        good_alias_1 = [x.text for x in self.annotator._nlp.tokenizer('Robby')]
        self.assertIn(good_alias_1, self.annotator._alias_lookup)

    def test_add_relations(self):
        """
        Tests that relations are properly added to ``Annotator._parents``
        """
        self.annotator._add_relations('Q1')
        self.assertIn('Q0', self.annotator._parents)

    def test_expand(self):
        """
        Tests that expanding a node adds its relations as well as notes where
        it was last seen.
        """
        loc = 0
        self.annotator._expand('Q1', loc)
        self.assertIn('Q1', self.annotator._last_seen)
        self.assertEqual(self.annotator._last_seen['Q1'], loc)
        self.assertIn('Q0', self.annotator._parents)

    def test_existing_id(self):
        """
        Test that annotator can extract stack of tokens with existing
        annotations (from WIKI/NEL/COREF step)
        """
        doc = self.annotator._json_to_doc(self.json_data, root_id='Q0')
        token_stack = deque(reversed(doc))
        start_length = len(token_stack)
        active =  token_stack.pop()
        match_stack = self.annotator._existing_id(active, token_stack)
        expected = deque(((doc[0], 'Q0'), (doc[1], 'Q0')))
        self.assertEqual(match_stack, expected)
        self.assertEqual(len(token_stack), start_length - 2)

    def test_unknown_id_no_match(self):
        """
        Test that annotator can identify stack of tokens with no
        annotations.
        """
        doc = self.annotator._json_to_doc(self.json_data, root_id='Q0')
        token_stack = deque(reversed(doc))
        token_stack.pop()  # Pop 'Robert'
        token_stack.pop()  # Pop 'Plant'
        # Check 'sings' is popped with no id
        start_length = len(token_stack)
        active = token_stack.pop()
        match_stack = self.annotator._unknown_id(active, token_stack)
        expected = deque(((doc[2], None),))
        self.assertEqual(match_stack, expected)
        self.assertEqual(len(token_stack), start_length - 1)

    def test_unknown_id_alias_match(self):
        # Assume Q1 and  Q2's aliases have been added to the lookup
        self.annotator._add_aliases('Q1')
        self.annotator._add_aliases('Q2')
        # 'Robert Lockwood Logan IV' should all get matched - the second token
        # should be associated with Q2 (blues musician Robert Lockwood) and
        # the fourth token should be associated with Q1.
        tokens = ['Robert', 'Lockwood', 'Logan', 'IV']
        doc = Doc(self.annotator._nlp.vocab, words=tokens)
        token_stack = deque(reversed(doc))
        active = token_stack.pop()
        match_stack = self.annotator._unknown_id(active, token_stack)
        expected = deque((
            (doc[0], None),
            (doc[1], 'Q2'),
            (doc[2], None),
            (doc[3], 'Q1')))
        self.assertEqual(match_stack, expected)
        self.assertEqual(len(token_stack), 0)

    def test_annotate_tokens(self):
        doc = self.annotator._json_to_doc(self.json_data, root_id='Q0')
        self.annotator._annotate_tokens(doc)
        self.assertEqual(doc[0]._.id, 'Q0')
        self.assertEqual(doc[0]._.source, 'WIKI')
        self.assertEqual(doc[0]._.parent_id, ['Q0'])
        self.assertEqual(doc[0]._.relation, ['@@NEW@@'])
        self.assertEqual(doc[4]._.id, 'Q3')
        self.assertEqual(doc[4]._.source, 'NEL')
        self.assertEqual(doc[4]._.parent_id, ['Q0'])
        self.assertEqual(doc[4]._.relation, ['plays_in'])
        self.assertEqual(doc[7]._.id, 'Q0')
        self.assertEqual(doc[7]._.source, 'COREF')
        self.assertEqual(doc[7]._.parent_id, ['Q0'])
        self.assertEqual(doc[7]._.relation, ['@@REFLEXIVE@@'])

    def test_serialize_annotations(self):
        doc = self.annotator._json_to_doc(self.json_data, root_id='Q0')
        self.annotator._annotate_tokens(doc)
        annotations = self.annotator._serialize_annotations(doc)
        expected = [
            {
                'source': 'WIKI',
                'id': 'Q0',
                'relation': ['@@NEW@@'],
                'parent_id': ['Q0'],
                'span': [0, 2]
            },
            {
                'source': 'NEL',
                'id': 'Q3',
                'relation': ['plays_in'],
                'parent_id': ['Q0'],
                'span': [4, 6]
            },
            {
                'source': 'COREF',
                'id': 'Q0',
                'relation': ['@@REFLEXIVE@@'],
                'parent_id': ['Q0'],
                'span': [7, 8]
            }
        ]
        self.assertEqual(annotations, expected)

    def test_add_wikilinks_match_aliases(self):
        """
        If `match_aliases` is enabled, then `Annotator._add_wikilinks`
        should add the aliases of wikilinks to `Annotator._alias_lookup`
        """
        # Set up
        annotator = Annotator(alias_db=self.alias_db,
                              relation_db=self.relation_db,
                              wiki_db=self.wiki_db,
                              match_aliases=True)
        tokens = flatten_tokens(self.json_data['tokens'])
        doc = Doc(annotator._nlp.vocab, words=tokens)

        # Test
        annotator._add_wikilinks(doc, self.json_data['entities'])
        self.assertIn(['Robert', 'Plant'], annotator._alias_lookup)

    def test_add_nel_unmatch(self):
        # Set up
        annotator = Annotator(alias_db=self.alias_db,
                              relation_db=self.relation_db,
                              wiki_db=self.wiki_db,
                              unmatch=True)
        tokens = flatten_tokens(self.json_data['tokens'])
        doc = Doc(annotator._nlp.vocab, words=tokens)
        wiki_ids = {'Q0'}

        # Test
        annotator._add_nel(doc, self.json_data['nel'], wiki_ids)
        # "Led Zeppelin" should not be linked, since it is 'unseen'
        self.assertNotEqual(doc[4]._.id, 'Q3')
        self.assertNotEqual(doc[5]._.id, 'Q3')
