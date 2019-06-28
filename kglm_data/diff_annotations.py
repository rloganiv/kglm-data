"""
Detect differences between different annotations of the same data.
"""
import argparse
from collections import defaultdict
import json
import logging


def default_factory():
    return [None, None]


def diff(data1, data2):
    token_ids = defaultdict(default_factory)
    for annotation in data1['annotations']:
        for i in range(*annotation['span']):
            token_ids[i][0] = annotation
    for annotation in data2['annotations']:
        for i in range(*annotation['span']):
            token_ids[i][1] = annotation

    joint_annotations = []
    for i, annotations in token_ids.items():
        joint_annotation = {}
        joint_annotation['span'] = [i, i+1]
        joint_annotation['annotations'] = annotations
        joint_annotations.append(joint_annotation)

    data = data1.copy()
    data['annotations'] = joint_annotations

    return data


def main(_):
    with open(FLAGS.file1, 'r') as f, open(FLAGS.file2, 'r') as g:
        for line1, line2 in zip(f, g):
            data1 = json.loads(line1)
            data2 = json.loads(line2)
            assert data1['tokens'] == data2['tokens'], 'Data mismatch'
            data = diff(data1, data2)
            data['aliases'] = (FLAGS.alias1, FLAGS.alias2)
            print(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str)
    parser.add_argument('file2', type=str)
    parser.add_argument('--alias1', type=str, default=None)
    parser.add_argument('--alias2', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL)

    main(_)

