"""
Constructs a key-value store mapping Wikidata IDs to a list of related Wikidata
entities
"""
import argparse
from collections import defaultdict
import csv
import json
import logging
import pickle
import re
import sys

from sqlitedict import SqliteDict

from kglm_data.util import generate_from_wikidump, load_allowed_entities

logger = logging.getLogger(__name__)


BAD_DATATYPES = ['external-id', 'url', 'commonsMedia', 'globecoordinate']
RE_PROPERTY = re.compile('(?<=http:\/\/www.wikidata.org\/entity\/)P\d+')


def load_allowed_properties(fname):
    """Loads a set of allowed properties from a csv file."""
    if fname is None:
        logger.info('Properties not restricted')
        return
    else:
        logger.info('Loading allowed properties from: "%s"', fname)
        allowed_properties = set()
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                logger.debug(row)
                match = RE_PROPERTY.search(row[0])
                if match is not None:
                    logger.debug('Found property "%s"', match.group(0))
                    allowed_properties.add(match.group(0))
        logger.info('%i allowed properties found', len(allowed_properties))
        return allowed_properties


def main(_):
    allowed_properties = load_allowed_properties(FLAGS.properties)
    allowed_entities = load_allowed_entities(FLAGS.entities)

    if FLAGS.in_memory:
        db = defaultdict(list)
    else:
        logger.info('Opening data file at: "%s"', FLAGS.db)
        db = SqliteDict(FLAGS.db, autocommit=True, journal_mode='OFF')

    if FLAGS.reverse and not FLAGS.in_memory:
        raise RuntimeError('Cannot add reverse relations to out-of-memory db '
                           '(takes too long).')

    for data in generate_from_wikidump(FLAGS.input):
        id = data['id']

        # Check if data pertains to a given entity
        if allowed_entities is not None:
            if id not in allowed_entities:
                continue

        claims = data['claims']
        properties = []
        for property, snaks in claims.items():

            # Check if property is allowed
            if allowed_properties is not None:
                if property not in allowed_properties:
                    continue

            for snak in snaks:

                # Check if top-level relation is allowed
                mainsnak = snak['mainsnak']
                if mainsnak['datatype'] in BAD_DATATYPES:
                    continue
                try:
                    value = mainsnak['datavalue']
                except KeyError:
                    continue

                # Seperate processing for monolingual text
                if mainsnak['datatype'] == 'monolingualtext':
                    # Only accept english strings...
                    if value['value']['language'] != 'en':
                        continue

                # If relation is between entities, check that tail entity is
                # allowed
                # if allowed_entities is not None:
                #     if mainsnak['datatype'] == 'wikibase-item':
                #         tail_id = mainsnak['datavalue']['value']['id']
                #         if tail_id not in allowed_entities:
                #             continue

                properties.append((property, value))

                # Next process qualifiers
                if 'qualifiers' in snak:

                    qualifiers = snak['qualifiers']

                    for qual_prop, qual_snaks in qualifiers.items():

                        qual_prop = property+':'+qual_prop
                        for qual_snak in qual_snaks:

                            # Check relation is allowed
                            if qual_snak['datatype'] in BAD_DATATYPES:
                                continue
                            try:
                                qual_value = qual_snak['datavalue']
                            except KeyError:
                                continue

                            # Seperate processing for monolingual text
                            if qual_snak['datatype'] == 'monolingualtext':
                                # Only accept english strings...
                                if qual_value['value']['language'] != 'en':
                                    continue
                            # If relation is between entities, check that tail
                            # entity is allowed
                            # if allowed_entities is not None:
                            #     if qual_snak['datatype'] == 'wikibase-item':
                            #         tail_id = qual_snak['datavalue']['value']['id']
                            #         if tail_id not in allowed_entities:
                            #             continue

                            properties.append((qual_prop, qual_value))

        logger.debug('Entity: %s', id)
        logger.debug('Properties: %s', properties)

        if FLAGS.in_memory:
            db[id].extend(properties)
        else:
            db[id] = properties

        # Optional: Add reverse links
        if FLAGS.reverse:
            logger.debug('Adding reverse links')
            for rel, tail in properties:
                if tail['type'] != 'wikibase-entityid':
                    continue
                tail_id = tail['value']['id']
                bw_prop = 'R:' + rel
                bw_value = {
                    'type': 'wikibase-entityid',
                    'value': {
                        'id': id
                    }
                }
                db[tail_id].append((bw_prop, bw_value))

    if FLAGS.in_memory:
        logger.info('Dumping')
        with open(FLAGS.db, 'wb') as f:
            pickle.dump(db, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--db', type=str, default='relation.db')
    parser.add_argument('-p', '--properties', type=str, default=None)
    parser.add_argument('-e', '--entities', type=str, default=None)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL)

    main(_)

