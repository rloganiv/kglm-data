import json
import logging
import os
import unittest

from spacy.tokens import Doc, Token

from src.annotate import Annotator

logger = logging.getLogger(__name__)


class TestAnnotator(unittest.TestCase):
    def setUp(self):
        alias_db = {
            'Q0': ['Robert Plant'],
            'Q1': ['Robert Lockwood Logan IV', 'Robby'],
            'Q2': ['Robert Lockwood']
        }
        wiki_db = {
            'robert_plant': 'Q0',
            'robert_lockwood_logan_iv': 'Q1',
            'robert_lockwood': 'Q2'
        }
        relation_db = {
            'Q0': [(('influenced_by', None),
                    {'type': 'wikibase-entityid', 'value':{'id':'Q2'}})],
            'Q1': [(('listens_to', None),
                    {'type': 'wikibase-entityid', 'value':{'id':'Q0'}})],
            'Q2': []
        }
        self.annotator = Annotator(alias_db, relation_db, wiki_db,
                                   distance_cutoff=5)
        with open('tests/fixtures/test.jsonl', 'r') as f:
            self.test_line = f.readline()
        json_data = json.loads(self.test_line.strip())
        self.test_doc = self.annotator._json_to_doc(json_data)

    def test_reset(self):
        self.annotator._reset()
        assert self.annotator._last_seen == {}
        assert self.annotator._parents == {}

    def test_propagate_ids(self):
        self.annotator._propagate_ids(self.test_doc)
        assert self.test_doc[7]._.id == 'Q0'

    def test_add_aliases(self):
        self.annotator._add_aliases('Q1')
        logger.debug(self.annotator._alias_lookup._root)
        good_alias_0 =  [x.text for x in self.annotator._nlp.tokenizer('Robert Lockwood Logan IV')]
        assert good_alias_0 in self.annotator._alias_lookup
        good_alias_1 =  [x.text for x in self.annotator._nlp.tokenizer('Robby')]
        assert good_alias_1 in self.annotator._alias_lookup

    def test_add_relations(self):
        self.annotator._add_relations('Q1')
        logger.debug(self.annotator._alias_lookup._root)
        good_alias =  [x.text for x in self.annotator._nlp.tokenizer('Robert Plant')]
        assert good_alias in self.annotator._alias_lookup
        assert 'Q0' in self.annotator._parents
