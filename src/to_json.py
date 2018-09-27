"""Converts the .tsv datasets to JSON for viewing"""

import argparse
import csv
import json
import logging
import sys

from sqlitedict import SqliteDict


logger = logging.getLogger(__name__)


class JsonWriter(object):
    def __init__(self, alias_db):

        self.link_count = 0
        self.word_count = 0
        self.no_parent_count = 0
        self.parent_count = 0

        self.alias_db = alias_db

        self.seen = set()
        self.last = dict()
        self.unique_comments = dict()
        self.text = ''
        self.attributes = []
        self.entities = []
        self.relations = []
        self.comments = []

    def update(self, row):
        start = len(self.text)
        end = start + len(row['text'])
        if row['z'] == 'True':
            if row['id'] is '':
                previous_entity = self.entities[-1]
                previous_entity[2][0][1] = end # Stupid hack to expand span
                self.entities[-1] = previous_entity
            else:
                self.link_count += 1
                id = row['id']
                parent_id = row['parent_id']
                # Get names
                if self.alias_db:
                    try:
                        entity_name = self.alias_db[id][0]
                    except KeyError:
                        entity_name = 'literal:%s' % id
                    except IndexError:
                        entity_name = id
                else:
                    entity_name = id
                # Add a new entity
                eid = 'E%i' % len(self.entities)
                entity = [eid, entity_name, [[start, end]]]
                self.entities.append(entity)
                # Add a new relation
                rid = 'R%i' % len(self.relations)
                if id in self.seen:
                    # !!! DO NOT RENDER @@REFLEXIVE@@ RELATIONS !!!
                    # relation = [rid, row['relation'], [['Tail', self.last[id]], ['Head', eid]]]
                    # self.relations.append(relation)
                    pass
                elif parent_id in self.seen:
                    self.parent_count += 1
                    relation = [rid, row['relation'], [['Tail', self.last[parent_id]], ['Head', eid]]]
                    self.relations.append(relation)
                else:
                    aid = 'A%i' % len(self.attributes)
                    attribute = [aid, 'Orphan', eid]
                    self.attributes.append(attribute)
                    self.no_parent_count += 1
                # Add a new comment
                if id in self.unique_comments:
                    comment = self.unique_comments[id]
                else:
                    comment = [eid, 'NodeProperties', {'id': id, 'rootId': parent_id, 'relation': row['relation']}]
                    self.unique_comments[id] = comment
                self.comments.append(comment)
                # Mark that this was the last time entity was seen
                self.seen.add(id)
                self.last[id] = eid
        self.text += row['text'] + ' '
        self.word_count += 1
        if row['text'] == '.':
            self.text += '\n'

    def to_dict(self):
        out = {
            'attributes': self.attributes,
            'text': self.text,
            'entities': self.entities,
            'relations': self.relations,
            'comments': self.comments
        }
        return out


def generate_chunks(reader):
    out = []
    prev_page_id = '0'
    for row in reader:
        if row['page_id'] == prev_page_id:
            out.append(row)
        else:
            yield(out)
            out = [row]
            prev_page_id = row['page_id']
    yield(out)


def main(_):
    FIELD_NAMES = ['page_id', 'line_id', 'text', 'z', 'id', 'relation', 'parent_id']
    link_count = 0
    word_count = 0
    no_parent_count = 0
    parent_count = 0

    if FLAGS.alias_db is not None:
        alias_db = SqliteDict(FLAGS.alias_db)
    else:
        alias_db = None

    with open(FLAGS.input, 'r') as f:
        logger.debug('Creating reader')
        reader = csv.DictReader(f, FIELD_NAMES, delimiter='\t')
        logger.debug('Generating chunks')
        for i, chunk in enumerate(generate_chunks(reader)):
            logger.debug('On chunk %i', i)
            logger.debug('Initializing new JsonWriter')
            json_writer = JsonWriter(alias_db=alias_db)
            for row in chunk:
                json_writer.update(row)
            fname = '%s.%i.json' % (FLAGS.prefix, i)
            logger.debug('Writing to file "%s"', fname)
            out = json_writer.to_dict()
            with open(fname, 'w') as f:
                json.dump(out, f)
            link_count += json_writer.link_count
            word_count += json_writer.word_count
            parent_count += json_writer.parent_count
            no_parent_count += json_writer.no_parent_count
    logger.info('Total number of words: %i', word_count)
    logger.info('Total number of linked words: %i', link_count)
    logger.info('Number of entities that come from nowhere: %i', no_parent_count)
    logger.info('Number of entities that come from other entities: %i', parent_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--alias_db', default=None)
    FLAGS, _ = parser.parse_known_args()

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    main(_)

