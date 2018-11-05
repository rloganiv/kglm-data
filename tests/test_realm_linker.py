import logging
import unittest

from src.annotate_article import EntityToken, EnhancedToken, RealmLinker, tokenizer

logger = logging.getLogger(__name__)


class TestRealmLinker(unittest.TestCase):
    def setUp(self):
        alias_db = {
            'Q0': ['Robert Plant'],
            'Q1': ['Robert Lockwood Logan IV'],
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
        self.linker = RealmLinker(alias_db, relation_db, wiki_db,
                                  distance_cutoff=5)

    def test_basic_match(self):
        sentence = 'Robert Plant'
        tokens = [EntityToken(x, None) for x in tokenizer.tokenize(sentence)]
        self.linker.instantiate('robert_plant')
        logger.debug(self.linker.reverse_aliases._root)
        output = self.linker.link(tokens)
        expected = [
            EnhancedToken('Robert', True, 'Q0', ('@@REFLEXIVE@@', None), 'Q0'),
            EnhancedToken('Plant', True)
        ]
        assert output == expected

    def test_relation(self):
        # Robert Plant is used to initialize the linker. Since he is
        # 'influenced_by' Robert Lockwood, we should be able to match this
        # entity as well.
        sentence = 'Robert Lockwood'
        tokens = [EntityToken(x, None) for x in tokenizer.tokenize(sentence)]
        self.linker.instantiate('robert_plant')
        logger.debug(self.linker.reverse_aliases._root)
        output = self.linker.link(tokens)
        expected = [
            EnhancedToken('Robert', True, 'Q2', ('influenced_by', None), 'Q0'),
            EnhancedToken('Lockwood', True)
        ]
        assert output == expected

    def test_complex_matches(self):
        # Robert Lockwood Logan IV should be linked due to reflexivity.
        # Robert Lockwood Logan III should not be linked.
        # Robert Plant should be linked due to 'listens_to' relation.
        # Robert Lockwood should be linked due to 'influenced_by' relation.
        sentence = 'Robert Lockwood Logan IV Robert Lockwood Logan '\
                'III Robert Plant Robert Lockwood'
        tokens = [EntityToken(x, None) for x in tokenizer.tokenize(sentence)]
        logger.debug(tokens)
        self.linker.instantiate('robert_lockwood_logan_iv')
        output = self.linker.link(tokens)
        expected = [
            EnhancedToken('Robert', True, 'Q1', ('@@REFLEXIVE@@', None), 'Q1'),
            EnhancedToken('Lockwood', True),
            EnhancedToken('Logan', True),
            EnhancedToken('IV', True),
            EnhancedToken('Robert'),
            EnhancedToken('Lockwood'),
            EnhancedToken('Logan'),
            EnhancedToken('III'),
            EnhancedToken('Robert', True, 'Q0', ('listens_to', None), 'Q1'),
            EnhancedToken('Plant', True),
            EnhancedToken('Robert', True, 'Q2', ('inflenced_by', None), 'Q0'),
            EnhancedToken('Lockwood', True)
        ]

    def test_distance_cutoff(self):
        sentence = 'blah blah blah blah blah blah Robert Plant'
        tokens = [EntityToken(x, None) for x in tokenizer.tokenize(sentence)]
        logger.debug(tokens)
        self.linker.instantiate('robert_lockwood_logan_iv')
        output = self.linker.link(tokens)
        for enhanced_token in output:
            assert enhanced_token.z == False

