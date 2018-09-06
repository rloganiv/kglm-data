#! /usr/bin/env python3.6
"""
Constructs a key-value stor mapping Wikidata IDs to a list of related Wikidata
entities """ 
import argparse
import json
import logging
import sys

from sqlitedict import SqliteDict


logger = logging.getLogger(__name__)
BAD_DATATYPES = ['external-id', 'url', 'commonsMedia', 'globecoordinate']


def main(_):
    logger.info('Opening data file at: "%s"', FLAGS.db)
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

            id = data['id']
            claims = data['claims']
            properties = []
            for property, snaks in claims.items():
                for snak in snaks:
                    # First get top-level relation
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
                    properties.append(((property, None), value))
                    # Next get qualifiers
                    if 'qualifiers' in snak:
                        qualifiers = snak['qualifiers']
                        for qual_prop, qual_snaks in qualifiers.items():
                            qual_prop = (property, qual_prop)
                            for qual_snak in qual_snaks:
                                if qual_snak['datatype'] in BAD_DATATYPES:
                                    continue
                                try:
                                    qual_value = qual_snak['datavalue']
                                except KeyError:
                                    continue
                                if qual_snak['datatype'] == 'monolingualtext':
                                    # Only accept english strings...
                                    if qual_value['value']['language'] != 'en':
                                        continue
                                properties.append((qual_prop, qual_value))

            logger.debug('Entity: %s', id)
            logger.debug('Properties: %s', properties)

            db[id] = properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='relation.db')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    main(_)

