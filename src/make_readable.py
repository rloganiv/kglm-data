"""
Adds human-readable labels to the generative story.
"""
import argparse
from collections import defaultdict
import json
import pickle

from sqlitedict import SqliteDict


def main(_):
    if FLAGS.in_memory:
        with open(FLAGS.alias_db, 'rb') as f:
            alias_db = pickle.load(f)
    else:
        alias_db = SqliteDict(FLAGS.alias_db, flag='r')

    def lookup(id):
        if id in alias_db:
            return alias_db[id][0]
        else:
            return ""

    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            annotations = data['annotations']
            readable_annotations = []

            for annotation in annotations:

                id = annotation['id']
                new_id = (id, lookup(id))

                parent_ids = annotation['parent_id']
                relations = annotation['relation']

                parent_strings = []
                relation_strings = []

                for parent_id, relation in zip(parent_ids, relations):

                    # Unpack relation string
                    if relation is None:
                        continue
                    readable = []
                    for piece in relation.split(':'):
                        if piece[0] == 'R':
                            readable.append('Reverse')
                        else:
                            readable.append(lookup(piece))
                    readable = ':'.join(readable)

                    parent_strings.append(lookup(parent_id))
                    relation_strings.append(readable)

                parent_ids = ', '.join(parent_ids)
                parent_strings = ', '.join(parent_strings)
                new_parent_id = (parent_ids, parent_strings)

                relations = ', '.join(relations)
                relation_strings = ', '.join(relation_strings)
                new_relation = (relations, relation_strings)

                # OLD APPROACH
                # new_relation = (relation, readable)
                # new_parent_id = (parent_id, lookup(parent_id))

                readable_annotation = annotation.copy()
                readable_annotation['id'] = new_id
                readable_annotation['relation'] = new_relation
                readable_annotation['parent_id'] = new_parent_id

                readable_annotations.append(readable_annotation)

            data['annotations'] = readable_annotations
            print(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    FLAGS, _ = parser.parse_known_args()

    main(_)

