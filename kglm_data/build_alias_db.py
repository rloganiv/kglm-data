"""
Construct a key-value store mapping Wikidata IDs to a list of labels/aliases
for the entity
"""
import argparse
from collections import defaultdict
import json
import logging
import pickle
import sys

from sqlitedict import SqliteDict

from util import generate_from_wikidump, load_allowed_entities, LOG_FORMAT

logger = logging.getLogger(__name__)


def main(_):
    allowed_entities = load_allowed_entities(FLAGS.entities)
    logger.info('Opening data file at: "%s"', FLAGS.db)

    if FLAGS.in_memory:
        db = defaultdict(list)
    else:
        db = SqliteDict(FLAGS.db, autocommit=True, journal_mode='OFF')

    for data in generate_from_wikidump(FLAGS.input):

        id = data['id']

        # Check whether entity appears in the dataset
        if allowed_entities is not None:
            is_entity = id[0] == 'Q'
            if id not in allowed_entities and is_entity:
                continue

        # Obtain aliases for entity
        aliases = []
        try:
            name = data['labels']['en']['value']
            aliases.append(name)
        except:
            logger.warning('No name found for entity "%s"', id)
        try:
            aliases.extend(x['value'] for x in data['aliases']['en'])
        except:
            # Not a warning since it is typical for there not to be
            # additional aliases for a given entity
            logger.debug('No additional aliases found for entity "%s"', id)

        # Handle given / family names ... THIS APPROACH IS FUCKING STUPID,
        # SINCE IT REQUIRES THIS SCRIPT BE RUN TWICE TO ACTUALLY WORK...
        # if 'P734' in data['claims']:
        #     name_id_list = data['claims']['P734']['datavalue']
        #     for name_id in name_id_list:
        #         try:
        #             name = db[name_id][0]
        #             logger.debug('Adding family name "%s"', name)
        #         except KeyError:
        #             pass
        # if 'P735' in data['claims']:
        #     name_id_list = data['claims']['P735']['datavalue']
        #     for name_id in name_id_list:
        #         try:
        #             name = db[name_id][0]
        #             logger.debug('Adding given name "%s"', name)
        #         except KeyError:
        #             pass

        if len(aliases)== 0:
            continue

        logger.debug('id: "%s" - aliases: %s', id, aliases)
        db[id] = aliases


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
    parser.add_argument('--db', type=str, default='alias.db')
    parser.add_argument('-e', '--entities', type=str, default=None)
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=LEVEL)

    main(_)

