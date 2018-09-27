#! /usr/bin/env  python3.6
"""
Construct a key-value store mapping Wikidata IDs to a list of labels/aliases
for the entity
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
            aliases = []
            try:
                name = data['labels']['en']['value']
                aliases.append(name)
            except:
                logger.warning('No name found for entity "%s"', id)
            try:
                aliases.extend([x['value'] for x in data['aliases']['en']])
            except:
                # Not a warning since it is typical for there not to be
                # additional aliases for a given entity
                logger.debug('No additional aliases found for entity "%s"', id)
            # Handle given / family names ... THIS APPROACH IS FUCKING STUPID,
            # SINCE IT REQUIRES THIS SCRIPT BE RUN TWICE TO ACTUALLY WORK...
            if 'P734' in data['claims']:
                name_id_list = data['claims']['P734']['datavalue']
                for name_id in name_id_list:
                    try:
                        name = db[name_id][0]
                        logger.debug('Adding family name "%s"', name)
                    except KeyError:
                        pass
            if 'P735' in data['claims']:
                name_id_list = data['claims']['P735']['datavalue']
                for name_id in name_id_list:
                    try:
                        name = db[name_id][0]
                        logger.debug('Adding given name "%s"', name)
                    except KeyError:
                        pass
            logger.debug('id: "%s" - aliases: %s', id, aliases)
            if len[aliases] == '0':
                continue
            db[id] = aliases
        db.commit()
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='alias.db')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    main(_)

