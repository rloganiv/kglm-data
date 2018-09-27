#! /usr/bin/env  python3.6
"""
Obtains a list of ClueWeb documents containing numerous entities from the
'Freebase Annotations of the ClueWeb Corporora' files.
"""

import argparse
import sys


def main(_):
    current_file = None
    entities = set()
    for line in sys.stdin:
        fields = line.strip().split()
        file = fields[0]
        entity = fields[-1]
        if file == current_file:
            entities.add(entity)
        else:
            if len(entities) >= FLAGS.n:
                sys.stdout.write(current_file + '\t'.join(entities) + '\n')
            current_file = file
            entities = set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    FLAGS, _ = parser.parse_known_args()

    main(_)

