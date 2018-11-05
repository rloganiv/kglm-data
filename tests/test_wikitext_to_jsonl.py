import logging
import unittest

import spacy
from sqlitedict import SqliteDict

from src.wikidump_to_jsonl import process

logger = logging.getLogger(__name__)


wikitext = """
Once men turned their thinking over to machines in the hope that
this would set them free. But that only permitted other men with
machines to enslave them. Walk without rhythm, lest you awaken the
great [[Sandworm (Dune)|Shai Hulud]].
"""
wikitext = wikitext.replace('\n', ' ').strip()

wiki_db = SqliteDict('/tmp/wiki.db')
wiki_db['sandworm_(dune)'] = 'Q0'

nlp = spacy.load('en_core_web_sm')

class TestProcess(unittest.TestCase):

    def test_process(self):
        out = process('test', wikitext, nlp, wiki_db)
        assert out['title'] == 'test'
        assert out['entities'] == [['Q0', 39, 41]]
        assert out['sentences'] == [0, 18, 30, 42]
        assert out['tokens'][-3:-1] == ['Shai', 'Hulud']

