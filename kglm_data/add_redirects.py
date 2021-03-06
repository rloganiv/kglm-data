"""
Adds redirects from wikipedia to wikidb (the database mapping titles to
wikidata entities).
"""
import argparse
import bz2
import logging
import pickle
import sys
from xml.etree import ElementTree
from xml.sax.saxutils import unescape

from sqlitedict import SqliteDict


logger = logging.getLogger(__name__)


xmlns = '{http://www.mediawiki.org/xml/export-0.10/}'


def extract_redirect(elem: ElementTree.Element) -> str:
    """Extracts redirects from a <page> element in the Wikipedia dump.

    Args:
        elem : ``xml.etree.ElementTree.Element``
            The <page> element to process.

    Returns:
        A tuple ``(from, to)`` containing the titles of the pages being
        redirected from and to if the page is a redirect, otherwise ``None``.
    """
    # Get page title
    title = elem.find(f'{xmlns}title')
    if title is None:
        logger.debug('<page> has no <title> element')
        return
    _from = title.text.replace(' ', '_').capitalize()
    # Check if page is a redirect
    redirect = elem.find(f'{xmlns}redirect')
    if redirect is None:
        logger.debug('<page> has no <redirect> element')
        return
    _to = redirect.attrib['title'].replace(' ', '_').capitalize()
    logger.debug('Redirect from "%s" to "%s"', _from, _to)
    return _from, _to


def main(_):
    logger.info('Opening database file at: "%s"', FLAGS.wiki_db)

    if FLAGS.in_memory:
        with open(FLAGS.wiki_db, 'rb') as f:
            wiki_db = pickle.load(f)
    else:
        wiki_db = SqliteDict(FLAGS.wiki_db, autocommit=True)

    with bz2.open(FLAGS.input, 'r') as f:
        tree = ElementTree.iterparse(f, events=('start', 'end'))
        root = None
        for event, elem in tree:
            if event == 'start':
                if root is None:
                    root = elem
                else:
                    continue
            if elem.tag == f'{xmlns}page':
                redirect = extract_redirect(elem)
                if redirect is None:
                    continue
                _from, _to = redirect
                logger.debug('Looking up "%s"', _to)
                try:
                    entity_id = wiki_db[_to]
                    logger.debug('Found id "%s"', entity_id)
                except KeyError:
                    logger.debug('Could not find "%s"', _to)
                    continue
                if _from in wiki_db:
                    logger.warning('"%s" already in database', _from)
                else:
                    wiki_db[_from] = entity_id
                elem.clear()
                root.clear()

    if FLAGS.in_memory:
        logger.info('Dumping')
        with open(FLAGS.wiki_db, 'wb') as f:
            pickle.dump(wiki_db, f)
    else:
        wiki_db.commit()

    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('wiki_db', type=str)
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=logging.INFO)

    main(_)

