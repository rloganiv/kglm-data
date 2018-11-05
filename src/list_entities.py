"""
Prints out a list of all of the (WIKI) entities in a dataset
"""
import argparse
import json

def main(_):
    entities = set()
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            for entity, *_ in data['entities']:
                entities.add(entity)
    for entity in entities:
        print(entity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    FLAGS, _ = parser.parse_known_args()

    main(_)

