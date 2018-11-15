"""
Gathers examples of relations being expressed.
"""
from typing import Any, Dict

import argparse
from collections import Counter, defaultdict
import json
import logging
import pickle
import random

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

BORING_RELATIONS = ['@@REFLEXIVE@@', '@@NEW@@']


def generate_instances(fname: str) -> Dict[str, Any]:
    """Generates instances from a JSON-lines file"""
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


def main(_):
    logger.info('Creating data structures')
    examples = defaultdict(list)
    relation_counts = Counter()
    generator_counts = Counter()
    length_counts = Counter()
    last_seen = dict()
    entity_string = '[[ %s | %s | %s ]]'

    logger.info('Reading alias db')
    with open(FLAGS.alias_db, 'rb') as f:
        alias_db = pickle.load(f)

    logger.info('Parsing annotations')
    for instance in generate_instances(FLAGS.input):
        tokens = [x for sent in instance['tokens'] for x in sent]  # Flat
        for annotation in instance['annotations']:
            relations = annotation['relation']
            parent_ids = annotation['parent_id']
            id = annotation['id']
            span = annotation['span']
            source = annotation['source']

            if len(relations) != len(set(relations)):
                raise RuntimeError('Duplicate relations detected')

            # If relation is interesting get the span of text
            for relation, parent_id in zip(relations, parent_ids):

                if relation in BORING_RELATIONS:
                    generator_counts[relation] += 1
                    continue
                else:
                    generator_counts['@@STORY@@'] += 1

                relation_strings = []
                for s in relation.split(':'):
                    if s in alias_db:
                        try:
                            alias = alias_db[s][0]
                        except IndexError:
                            alias = s
                        relation_strings.append(alias)
                    else:
                        relation_strings.append(s)
                    relation = ' : '.join(relation_strings)

                try:
                    parent_name = alias_db[parent_id][0]
                except IndexError:
                    parent_name = parent_id

                try:
                    name = alias_db[id][0]
                except IndexError:
                    name = id

                relation_counts[relation] += 1

                if parent_id in last_seen:
                    # Parent span info
                    parent_span = last_seen[parent_id]['span']
                    parent_source = last_seen[parent_id]['source']
                    # Span length
                    start = parent_span[0]
                    end = span[1]
                    length = end - start
                    # Update length counter
                    length_counts[length] += 1

                    if length < FLAGS.l:
                        # Format parent entity string
                        parent_tokens = ' '.join(tokens[parent_span[0]:parent_span[1]])
                        parent_string = entity_string % (parent_tokens, parent_name, parent_source)
                        # Format child entity string
                        child_tokens = ' '.join(tokens[span[0]:span[1]])
                        child_string = entity_string % (child_tokens, name, source)
                        # Add excerpt
                        excerpt = ' '.join(tokens[parent_span[1]:span[0]])
                        excerpt = ' '.join([parent_string, excerpt, child_string])
                        examples[relation].append(excerpt)

            # Update last seen with previous annotation
            last_seen[id] = annotation

    # Length CDF
    logger.info('Writing length CDF')
    with open(FLAGS.prefix + '.length_cdf.tsv', 'w') as f:

        length_list = sorted(length_counts.items(), key=lambda x: x[0])
        accumulated = 0.0
        total = sum(x[1] for x in length_list)

        f.write('Length\tCDF\n')
        for length, count in length_list:
            accumulated += count
            f.write('%i\t%0.4f\n' % (length, accumulated / total))

    # Relation Histogram
    logger.info('Writing relation counts')
    with open(FLAGS.prefix + '.relation_counts.tsv', 'w') as f:
        f.write('Relation\tCount\n')
        for tuple in generator_counts.most_common():
            f.write('%s\t%i\n' % tuple)
        for tuple in relation_counts.most_common():
            f.write('%s\t%i\n' % tuple)

    # Relation excerpts
    logger.info('Writing excerpts')
    with open(FLAGS.prefix + '.excerpts.txt', 'w') as f:
        for relation_tuple, _ in relation_counts.most_common():
            f.write('%s\n' % (relation_tuple,))
            for i, example in enumerate(examples[relation_tuple]):
                f.write('\t%i\t%s\n' % (i, example))
                if i == FLAGS.n:
                    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('prefix', type=str)
    parser.add_argument('-n', type=int, default=None,
                        help='Number of excerpts to print')
    parser.add_argument('-l', type=int, default=None,
                        help='Length cutoff')
    parser.add_argument('--alias_db', type=str, default='./data/alias.pkl')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL)

    main(_)

