"""
Computes summary statistics of KGLM dataset.
"""
import argparse
from collections import Counter
import json
import logging

from sqlitedict import SqliteDict

logger = logging.getLogger(__name__)


class Stats():
    def __init__(self):
        self._prop_counts = Counter()
        self._p_source_given_new = Counter()
        self._n = 0
        self._n_entities = 0
        self._n_tokens = 0
        self._n_entity_tokens = 0
        self._total_annotations = 0
        self._new_entities = 0
        self._related_entities = 0
        self._reflexive_entities = 0
        self._n_wiki = 0
        self._n_nel = 0
        self._n_coref = 0
        self._n_kg = 0

    def update(self, data):
        self._n += 1
        self._n_tokens += sum(len(x) for x in data['tokens'])
        annotations = data['annotations']
        for annotation in annotations:
            self._n_entities += 1
            self._total_annotations += 1
            relation = annotation['relation']
            if relation == ['@@NEW@@']:
                self._new_entities += 1
                self._p_source_given_new[annotation['source']] += 1
            elif relation == ['@@REFLEXIVE@@']:
                self._reflexive_entities += 1
            else:
                self._related_entities += 1
            #    self._prop_counts[relation] += 1
            span = annotation['span']
            span_length = span[1] - span[0]
            self._n_entity_tokens += span_length
            source = annotation['source']
            if source == 'WIKI':
                self._n_wiki += 1
            elif source == 'NEL':
                self._n_nel += 1
            elif source == 'COREF':
                self._n_coref += 1
            elif source == 'KG':
                self._n_kg += 1

    def log(self):
        print('Tokens = %i' % self._n_tokens)
        print('Total annotations = %i' % self._total_annotations)
        print('Avg. annotations / page = %0.4f' %
              (self._total_annotations / self._n))
        print('P(entity token) = %0.4f' %
              (self._n_tokens / self._n_entity_tokens))
        print('P(@@REFLEXIVE@@) = %0.4f' %
              (self._reflexive_entities / self._total_annotations))
        print('P(new entity comes from KG) = %0.4f' %
              (self._related_entities / (self._new_entities + self._related_entities)))
        print('P(WIKI) = %0.4f' % (self._n_wiki / self._n_entities))
        print('P(NEL) = %0.4f' % (self._n_nel / self._n_entities))
        print('P(COREF) = %0.4f' % (self._n_coref / self._n_entities))
        print('P(KG) = %0.4f' % (self._n_kg / self._n_entities))

        total = sum(self._p_source_given_new.values())
        for source, count in self._p_source_given_new.items():
            print('P(%s|@@NEW@@) = %0.4f' % (source, count / total))


def main(_):
    stats = Stats()
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            stats.update(data)
    stats.log()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    logging.basicConfig(level=LEVEL)

    main(_)

