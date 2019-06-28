#! /usr/bin/env  python3.6
"""
Construct a key-value store mapping Wikipedia labels to WikiData entity ids.
"""
import argparse
import json
import logging
import pickle
import sys

from sqlitedict import SqliteDict

from kglm_data.util import format_wikilink, generate_from_wikidump, \
    load_allowed_entities, LOG_FORMAT

logger = logging.getLogger(__name__)


def main(_):
    if FLAGS.in_memory:
        db = dict()
    else:
        db = SqliteDict(FLAGS.db, autocommit=True, journal_mode='OFF')

    for data in generate_from_wikidump(FLAGS.input):

        id = data['id']

        try:
            wikilink = data['sitelinks']['enwiki']['title']
        except KeyError:
            logger.debug('No enwiki title found for entity "%s"', id)
            continue
        else:
            wikilink = format_wikilink(wikilink)

        logger.debug('id: "%s" - enwiki title: "%s"', id, wikilink)
        if wikilink in db:
            logger.warning('Collision for enwiki title: "%s"', wikilink)

        db[wikilink] = id

    if FLAGS.in_memory:
        logger.info('Dumping')
        with open(FLAGS.db, 'wb') as f:
            pickle.dump(db, f)
    else:
        db.commit()

    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--db', type=str, default='wiki.db')
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=LEVEL)

    main(_)

