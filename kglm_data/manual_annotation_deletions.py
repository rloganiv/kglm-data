"""
Manual annotation deletions
"""
import argparse
import json
import re


BAD_SPANS = [['.'], ['"']]
CUTOFF = 30
POINT_IN_TIME = re.compile('.*P585$')


def main(args):
    with open(args.dataset, 'r') as f:
        for line in f:
            data = json.loads(line)
            flat_tokens = [t for s in data['tokens'] for t in s]
            clean_annotations = []
            last_seen = dict()
            for annotation in data['annotations']:
                start, end = annotation['span']
                last_seen[annotation['id']] = end
                if flat_tokens[start:end] in BAD_SPANS:
                    continue
                new_relation = []
                new_parent_id = []
                for relation, parent_id in zip(annotation['relation'],
                                               annotation['parent_id']):
                    if POINT_IN_TIME.match(relation):
                        continue
                    if parent_id not in last_seen:
                        continue
                    if start - last_seen[parent_id] > CUTOFF:
                        continue
                    new_relation.append(relation)
                    new_parent_id.append(parent_id)
                if len(new_relation) == 0:
                    continue
                clean_annotation = annotation.copy()
                clean_annotation['relation'] = new_relation
                clean_annotation['parent_id'] = new_parent_id
                clean_annotations.append(clean_annotation)
            new_data = data.copy()
            new_data['annotations'] = clean_annotations
            print(json.dumps(new_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args, _ = parser.parse_known_args()

    main(args)

