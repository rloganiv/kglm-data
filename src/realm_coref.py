"""
Custom predictor for AllenNLPs coref model which allows windowing.
"""
from typing import Any, Dict, List

import argparse
from itertools import accumulate
import json
import logging
from queue import Queue
from threading import Thread

import requests
from simplejson.errors import JSONDecodeError
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _window(l, size=3):
    n = len(l)
    for i in range(n-size+1):
        yield l[i:i+size]


def _add_offset(x, offset):
    if isinstance(x, int):
        return x + offset
    elif isinstance(x, list):
        updated = []
        for element in x:
            updated.append(_add_offset(element, offset))
        return updated
    else:
        raise TypeError('Applied ``_add_offset()`` to something other than a '
                        'list of ints')


def _tuplify_set(x):
    if not isinstance(x, set):
        return x
    else:
        return tuple(_tuplify_set(elt) for elt in x)


def _min_span(cluster):
    return min(x[0] for x in cluster)

def _merge_clusters(all_clusters):
    mapping = dict()
    for clusters in all_clusters:
        for cluster in clusters:
            cluster = set(tuple(x) for x in cluster)
            # Merge existing clusters into current cluster
            for span in cluster.copy():
                if span in mapping:
                    cluster.update(mapping[span])
            # Update all spans to point at current cluster
            for span in cluster:
                mapping[span] = cluster
    # The merged clusters are the unique values in ``mapping``. In order to use
    # the ``set()`` we need the clusters to be hashable, so we turn them into
    # tuples using ``_tuplify_set``.
    unique_clusters = list(set(_tuplify_set(x) for x in mapping.values()))
    unique_clusters = sorted(unique_clusters, key=_min_span)
    return list(unique_clusters)


class CoreNLPCorefPredictor:
    def __init__(self,
                 address: str = 'http://localhost:9000/',
                 n_threads: int = 16):
        self._address = address
        self._params = {
            'properties': '{"annotators": "coref", "outputFormat": "json", '
                          ' "tokenize.whitespace": true, "ssplit.eolonly": true}'
        }
        self._headers = {'charset': 'utf-8'}
        self._n_threads = n_threads

    def predict_json(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        queue = Queue()
        offset = 0
        logger.debug('Enqueuing tasks')
        for token_window in _window(inputs['tokens']):
            logger.debug('Offset: %s, Tokens: %s', offset, token_window)
            queue.put((token_window, offset))
            offset += len(token_window[0])

        logger.debug('Starting threads')
        threads = []
        clusters = []
        for _ in range(self._n_threads):
            thread = Thread(target=self.predict_instance,
                            args=(queue, clusters))
            threads.append(thread)
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        logger.debug('Joining threads')

        inputs['clusters'] = _merge_clusters(clusters)

        return inputs

    def predict_instance(self, queue, clusters):
        """
        Submit a single instance to the CoreNLP server to get coref
        annotations.

        Args:
            queue : ``Queue``
                A queue of (token, offset) tuples to be processed.
            clusters : ``List``
                A list to store extracted mention clusters
        """
        while True:

            if queue.empty():
                break

            tokens, offset = queue.get()
            data = '\n'.join(' '.join(x) for x in tokens)

            logger.debug('Submitting following data for prediction: "%s"', data)
            response = requests.post(self._address,
                                     data=data.encode('utf-8'),
                                     params=self._params,
                                     headers=self._headers)

            logger.debug('Response URL: %s', response.url)

            try:
                response_json = response.json()
            except JSONDecodeError:
                logger.warning('No response for data: "%s"', data)
                continue

            coref = response_json['corefs']
            instance_clusters = []
            sent_starts = [0, *list(accumulate(len(x) for x in tokens))[:-1]]
            for cluster in coref.values():
                spans = []
                for x in cluster:
                    start = x['startIndex'] + sent_starts[x['sentNum'] - 1] - 1
                    end = x['endIndex'] + sent_starts[x['sentNum'] - 1] - 2
                    spans.append([start, end])
                instance_clusters.append(spans)

            instance_clusters = _add_offset(instance_clusters, offset)
            logger.debug('Extracted clusters: %s', instance_clusters)
            clusters.append(instance_clusters)


def main(_):
    logger.info('Starting predictor w/ %i threads', FLAGS.n_threads)
    coref_predictor = CoreNLPCorefPredictor(n_threads=FLAGS.n_threads)
    with open(FLAGS.input, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            try:
                out = coref_predictor.predict_json(data)
            except Exception as e:
                logger.warning('Error %s occurred processing: %s', e, data)
            else:
                print(json.dumps(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(_)

