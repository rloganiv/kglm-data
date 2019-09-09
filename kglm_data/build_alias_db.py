"""
Construct a key-value store mapping Wikidata IDs to a list of labels/aliases
for the entity
"""
import argparse
from collections import defaultdict
import logging
import pickle
from pathlib import Path
import re

from sqlitedict import SqliteDict

from kglm_data.util import generate_from_wikidump, load_allowed_entities, LOG_FORMAT, RE_ENTITY

logger = logging.getLogger(__name__)


def main(_):
    allowed_entities = load_allowed_entities(FLAGS.entities)
    logger.info('Opening data file at: "%s"', FLAGS.db)

    if FLAGS.in_memory:
        db = defaultdict(list)
    else:
        db = SqliteDict(FLAGS.db, autocommit=True, journal_mode='OFF')

    for data in generate_from_wikidump(FLAGS.input):

        wiki_id = data['id']

        # Check whether entity appears in the dataset
        if allowed_entities is not None:
            is_entity = re.match(RE_ENTITY, wiki_id)
            if wiki_id not in allowed_entities and is_entity:
                continue

        # Obtain aliases for entity
        aliases = []
        try:
            name = data['labels'][FLAGS.language]['value']
            aliases.append(name)
        except:
            logger.warning('No name found for entity "%s"', wiki_id)
        try:
            aliases.extend(x['value'] for x in data['aliases'][FLAGS.language])
        except:
            # Not a warning since it is typical for there not to be
            # additional aliases for a given entity
            logger.debug('No additional aliases found for entity "%s"', wiki_id)

        if len(aliases) == 0:
            continue

        logger.debug('id: "%s" - aliases: %s', wiki_id, aliases)
        db[wiki_id] = aliases

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
    parser.add_argument('-e', '--entities', type=Path, default=None)
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--language', type=str, default="en")
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=LEVEL)

    main(_)

