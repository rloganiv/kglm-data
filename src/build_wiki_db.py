#! /usr/bin/env  python3.6
"""
Construct a key-value store mapping Wikipedia labels to WikiData entity ids.
"""

import argparse
import json
import logging
import sys

from sqlitedict import SqliteDict


logger = logging.getLogger(__name__)


def main(_):
    logger.info('Opening database file at: "%s"', FLAGS.db)
    with SqliteDict(FLAGS.db, autocommit=True) as db:
        for line in sys.stdin:
            # Read JSON data into Python object
            if line[0] == '[':
                line = line[1:]
            elif line[-1] == ']':
                line = line[:-1]
            else:
                line = line[:-2]
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning('Could not decode line to JSON:\n"%s"\n', line)
                continue

            # Parse relevant fields
            id = data['id']
            try:
                wiki = data['sitelinks']['enwiki']['title']
                wiki = wiki.replace(' ', '_')
                try:
                    wiki = wiki[0].capitalize() + wiki[1:]
                except IndexError:
                    logger.warning('IndexError occured for title "%s"', wiki)
                    wiki = wiki.capitalize()
            except:
                logger.debug('No enwiki title found for entity "%s"', id)
                continue
            logger.debug('id: "%s" - enwiki title: "%s"', id, wiki)
            if wiki in db:
                logger.warning('Collision for enwiki title: "%s", ' \
                               'existing: "%s", new: "%s"', wiki, db[wiki], id)
            db[wiki] = id
        db.commit()
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='wiki.db')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    main(_)

