import argparse
import json
import numpy as np

from sqlitedict import SqliteDict


def convert(obj):
    obj['tokens'] = [x.replace('\n', ' ') for x in obj['tokens']]
    lengths = [0] + [len(x)+1 for x in obj['tokens']]
    indices = np.cumsum(lengths)
    text = ' '.join(x for x in obj['tokens'])
    text = ''.join([i if j % 100 else i + '\n' for j, i in enumerate(text)])
    last_seen = dict()
    entities = []
    relations = []
    attributes = []
    for annotation in obj['annotations']:
        [id, relation, parent_id], span = annotation
        start = int(indices[span[0]])
        end = int(indices[span[1]])
        try:
            entity_name = alias_db[id][0]
        except KeyError:
            entity_name = 'literal:%s' % id
        except IndexError:
            entity_name = id
        # Add a new entity
        eid = 'E%i' % len(entities)
        entity = [eid, entity_name, [[start, end]]]
        entities.append(entity)
        # Add a new relation
        rid = 'R%i' % len(relations)
        if id in last_seen:
            pass
        elif parent_id in last_seen:
            relation = [rid, relation, [['Tail', last_seen[parent_id]], ['Head', eid]]]
            relations.append(relation)
        else:
            aid = 'A%i' % len(attributes)
            attribute = [aid, 'Orphan', eid]
            attributes.append(attribute)
        last_seen[id] = eid
    out = {
        'attributes': attributes,
        'text': text,
        'entities': entities,
        'relations': relations
    }
    return out


def main(_):
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            out = convert(data)
            print(out)
            with open('viewer/data/' + data['title'] + '.json', 'w') as g:
                json.dump(out, g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    FLAGS, _ = parser.parse_known_args()

    alias_db =  SqliteDict(FLAGS.alias_db, flag='r')

    main(_)

