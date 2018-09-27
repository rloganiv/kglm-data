"""
Extract the text from a wikipedia dump and serialize to a (large) jsonl file.
"""

import argparse
import bz2
from collections import UserDict
import json
import logging
import os
import re
import sys
from typing import Tuple
from xml.etree import ElementTree

import mwparserfromhell as wiki
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from sqlitedict import SqliteDict

logger = logging.getLogger(__name__)


BAD_NODES = ['ref', 'table']
BAD_SECTIONS = ['see also', 'further reading', 'references', 'external links']
RE_WHITESPACE = re.compile(r'[\n\r\s]+')


def page_generator(path: str) -> Tuple[str]:
    """Generates page titles and wikitext from an XML dump of Wikipedia.

    Args:
        path : ``str``
            Path to the wikimedia dump.

    Yields:
        title : ``str``
            The page title.
        wikitext : ``str``
            The page wikitext.
    """
    xmlns = '{http://www.mediawiki.org/xml/export-0.10/}'
    with bz2.open(path, 'r') as f:
        tree = ElementTree.iterparse(f, events=('start', 'end'))
        root = None
        for event, elem in tree:
            # Need to get root element, so we can clear it everytime we see a
            # page. Otherwise the document tree will grow too large to fit in
            # memory.
            if event ==  'start':
                if root is None:
                    root = elem
                else:
                    continue
            # AFAIK articles (as opposed to redirects / docs / etc) are
            # distinguished by:
            #   1. Not having a `redirect` node.
            #   2. Having a `ns` node with value 0.
            if elem.tag == f'{xmlns}page':
                title = elem.find(f'{xmlns}title')
                wikitext = elem.find(f'.//{xmlns}text')
                ns = elem.find(f'{xmlns}ns')
                redirect = elem.find(f'{xmlns}redirect')
                logger.debug(ns.text)
                if redirect is None and ns.text == '0':
                    try:
                        yield title.text, wikitext.text
                    except AttributeError:
                        pass
                elem.clear()
                root.clear()


def clean_wikitext(wikitext: str,
                   max_newline_freq: float = 0.01) -> str:
    """Strips most of the markup and clutter (e.g. templates, references,
    tables, lists) from a string containing wikitext.

    Args:
        wikitext : ``str``,
            The wikitext to process.
        max_newline_freq : ``float``, (default = 0.01)
            Maximum frequency of newline characters allowed in a section. If
            newline frequency exceeds this number then section will be omitted
            from the cleaned text. This is used as a heuristic for identifying
            lists.

    Returns:
         The cleaned wikitext.
    """
    # First pass - remove unwanted nodes from the document
    wikicode = wiki.parse(wikitext, skip_style_tags=True)
    nodes = []
    for node in wikicode.nodes:
        if isinstance(node, wiki.nodes.Heading):
            if node.title.lower().strip() in BAD_SECTIONS:
                break
            nodes.append(str(node))
        elif isinstance(node, wiki.nodes.Wikilink):
            if 'File:' in node.title or 'Image:' in node.title:
                continue
            nodes.append(str(node))
        elif isinstance(node, wiki.nodes.Template):
            continue
        elif isinstance(node, wiki.nodes.Tag):
            if node.tag.lower() in BAD_NODES:
                continue
        else:
            stripped = node.__strip__(normalize=True, collapse=True,
                                      keep_template_params=False)
            if stripped:
                nodes.append(str(stripped))
    stripped = ''.join(nodes).strip('\n')

    # Iterate over sections and use newline frequency as a heuristic for
    # filteriing out lists.
    wikicode = wiki.parse(stripped, skip_style_tags=True)
    nodes = []
    for section in wikicode.get_sections(flat=True, include_headings=False):
        text = str(section)
        nlines = text.count('\n')
        nchar = len(text)
        try:
            newline_density = nlines / nchar
        except ZeroDivisionError:
            continue
        if newline_density <= 0.01:
            nodes.append(text)
    stripped = ''.join(nodes).strip('\n')

    # Lastly, some minor text reformatting.
    while '\n\n\n' in stripped:
        stripped = stripped.replace('\n\n\n', '\n\n')
    stripped = stripped.replace("'''", "")
    stripped = stripped.replace("''", "")
    return stripped


def process(title: str,
            wikitext: str,
            nlp: Language,
            wiki_db: SqliteDict,
            batch_size: int = 128,
            n_threads: int = 1) -> str:
    """Tokenizes text, maps wikilinks to wikidata entities, and constructs
    output dictionary.

    Args:
        title: ``str``
            Article title.
        wikitext : ``str``
            Cleaned article wikitext.
        nlp : ``Language``
            SpaCy model used to tokenize text / split into sentences.
        wiki_db : ``SqliteDict``
            Lookup used to map Wikilinks to Wikidata Ids.
        batch_size : ``int`` (default=128)
            Batch size for SpaCy to use.
        n_threads : ``int`` (default=1)
            Number of threads for SpaCy to use.

    Returns:
        A dictionary of the form:

        {
            'tokens': [tokenized document text],
            'sentences': [indices of sentence starts, ..., len(document)],
            'entities':
                [
                    [id, start index, end index],
                    ...
                    [id, start index, end index]
                ]
        }

        Note: Document length added to end of 'sentence_boundaries' for
        convenience. This way sentences can be generated by:

            for start, end in zip(sentences[:-1], sentences[1:]):
                sentence = document[start:end]

    """
    # IDEA: Tokenize using SpaCy, keep track of tokens to identify labeled
    # spans (e.g. we just need [id, start index, end index] no need to
    # interact with SpaCy directly). Once all of the tokens are in a single
    # list, create a doc and apply the rest of the pipeline to get sentence
    # boundaries.
    wikitext = wikitext.strip()
    wikicode = wiki.parse(wikitext, skip_style_tags=True)
    words = []
    spaces = []
    entities = []

    texts = []
    ids = []
    for node in wikicode.nodes:
        id = None
        if isinstance(node, wiki.nodes.Wikilink):
            node_title = str(node.title)
            key = node_title.replace(' ', '_')
            try:
                key = key[0].capitalize() + key[1:]
            except IndexError:
                if len(key) == 1:
                    key = key[0].capitalize()
                else:
                    key = None
            if key in wiki_db:
                id = wiki_db[key]
            if node.text is not None:
                text = str(node.text)
            else:
                text = node_title
        else:
            text = node.__strip__(normalize=True, collapse=True,
                                  keep_template_params=False)
            text = str(text)
        texts.append(text)
        ids.append(id)

    for id, doc in zip(ids, nlp.tokenizer.pipe(texts, batch_size, n_threads)):
        start = len(words)
        tokens = [token for token in doc if not RE_WHITESPACE.search(token.text)]
        words.extend([token.text for token in tokens])
        spaces.extend([token.whitespace_  != '' for token in tokens])
        end = len(words)
        if id is not None:
            entities.append([id, start, end])
    assert len(words) == len(spaces), 'mismatching lengths: `words` and `spaces`'

    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    tokens = [[token.text for token in sentence] for sentence in doc.sents]

    if len(tokens) == 0:
        return

    out = {
        'title': title,
        'tokens': tokens,
        'entities': entities
    }

    return out


def main(_):
    logger.info('Loading SpaCy model')
    nlp = spacy.load('en_core_web_sm', disable=['ner'])

    logger.info('Loading wiki database from "%s"', FLAGS.wiki_db)
    wiki_db = SqliteDict(FLAGS.wiki_db, flag='r')
    assert wiki_db.flag == 'r', '`wiki_db` must be opened as read-only'

    for i, (title, wikitext) in enumerate(page_generator(FLAGS.input)):
        if not i % 10000:
            logger.info('On article %i', i)
        logger.debug('Processing article "%s"', title)
        title = title.lower().replace(' ', '_')
        wikitext = clean_wikitext(wikitext)
        instance = process(title, wikitext, nlp, wiki_db, n_threads=FLAGS.n_threads)
        if instance is not None:
            print(json.dumps(instance))
    wiki_db.close()

    logger.info('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to Wikipedia dump')
    parser.add_argument('--wiki_db', type=str, default='data/wiki.db',
                        help='Path to database mapping article titles to '
                             'wikidata ids')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug statements')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='Number of threads to use')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    logging.basicConfig(stream=sys.stderr,
                        level=LEVEL,
                        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                        datefmt="%H:%M:%S")

    main(_)

