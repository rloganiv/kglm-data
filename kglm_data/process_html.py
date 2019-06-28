"""
Strips text and WikiLinks from HTML output by the MediaWiki parser.
"""
from typing import Any, Dict

import argparse
import json
import logging
import pickle
import re

from bs4 import BeautifulSoup, Comment
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from sqlitedict import SqliteDict

from kglm_data.util import format_wikilink

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


RE_WHITESPACE = re.compile(r'[\n\r\s]+')
RE_HEADER = re.compile(r'^h[1-6]')


def generate_instances(input: str) -> Dict[str, Any]:
    """Generates instances from an input JSON-lines file"""
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            yield data


def clean_soup(root: BeautifulSoup) -> None:
    """Cleans the parsed HTML tree.

    This is done in the following steps:
        - Remove all unwanted elements and their children from the tree.
        - Replacing all math elements with a <formula> token.
        - Clearing formatting
    """
    # Remove all top-level unwanted elements
    unwanted_tags = ['div', 'table', 'ul', 'style']
    for tag in unwanted_tags:
        for branch in root(tag, recusive=False):
            branch.decompose()
    # Remove all reference tags
    for reference in root.select('.reference'):
        reference.decompose()
    # Remove all 'edit section' spans from headings
    for edit in root.select('span.mw-editsection'):
        edit.decompose()
    # Remove any page elements which are not rendered
    for invisible in root.select('.noprint,.mw-empty-elt'):
        invisible.decompose()
    # Comments need to be handled seperately
    for comment in root(string = lambda text:isinstance(text, Comment)):
        comment.extract()
    # Math is typically rendered two ways: inline (as a <span>), and as a
    # seperate line (as a <dl><dd><span>). Unfortunately, math can also just be
    # italicized text
    for equation in root.select('span.mwe-math-element'):
        equation.replace_with('__LATEX_FORMULA__')
    # We can clear formatting by using replace_with_children
    format_tags = ['i', 'span', 'dl', 'dt']
    for tag in format_tags:
        for branch in root(tag):
            branch.replaceWithChildren()


def process(title: str,
            root: BeautifulSoup,
            wiki_db: SqliteDict,
            nlp: Language) -> Dict[str, Any]:
    """Processes HTML tree into an annotated text object."""
    ids = []
    text = []
    try:
        title_id = wiki_db[format_wikilink(title)]
    except KeyError:
        logger.warning('No wiki entity associated with: %s', title)

    # First we build up a list of (text, id) pairs
    def _recursion(node):
        # If node is a link, make sure text is annotated with the corresponding
        # title attribute. NOTE: title is an attribute of the <a> tag. Don't
        # get confused with the page title.
        if node.name == 'a':
            try:
                key = format_wikilink(node['title'])
            except KeyError:
                key = None
            if key in wiki_db:
                id = wiki_db[key]
            else:
                id = None
            ids.append(id)
            text.append(node.text)
        # TODO: Figure out what we want to do with section titles
        elif RE_HEADER.search(str(node.name)):
            pass
        elif node.name == 'b' and len(text) < 50:
            logger.debug('Associating %s w/ title', node.text)
            ids.append(title_id)
            text.append(node.text)
        # Otherwise, continue to recurse
        else:
            if hasattr(node, 'children'):
                for child in node.children:
                    _recursion(child)
            # If we've hit a leaf (that isn't a link) then append the text
            else:
                ids.append(None)
                text.append(str(node))

    _recursion(root)

    # Next we'll tokenize
    words = []
    spaces = []
    entities = []
    for id, doc in zip(ids, nlp.tokenizer.pipe(text, FLAGS.batch_size, FLAGS.n_threads)):
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

    out = {
        'title': title,
        'tokens': tokens,
        'entities': entities
    }

    return out


def main(_):
    with open(FLAGS.wiki_db, 'rb') as f:
        wiki_db = pickle.load(f)
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    for instance in generate_instances(FLAGS.input):
        soup = BeautifulSoup(instance['html'])
        root = soup.div
        clean_soup(root)
        try:
            processed = process(instance['title'], root, wiki_db, nlp)
        except NameError:
            continue
        print(json.dumps(processed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--wiki_db', type=str,
                        default='./data/wiki.pkl')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_threads', type=int, default=32)
    parser.add_argument('--j', type=int, default=32)
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL)

    main(_)

