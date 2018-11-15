"""
Measures fan-in and fan-out of relations.
"""
import argparse
from collections import defaultdict
import csv
import json
import logging
import pickle
from statistics import mean, stdev, StatisticsError

logger = logging.getLogger(__name__)


def fan_stats(edges):
    _avg = mean(len(x) for x in edges.values())
    try:
        _std = stdev(len(x) for x in edges.values())
    except StatisticsError:
        _std = 0.0
    _max = max(len(x) for x in edges.values())
    _min = min(len(x) for x in edges.values())
    return _avg, _std, _max, _min


def readable(relation, alias_db):
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
    return relation




def main(_):

    logger.info('Loading pickled data...')
    with open(FLAGS.alias_db, 'rb') as f:
        alias_db = pickle.load(f)

    with open(FLAGS.relation_db, 'rb') as f:
        relation_db = pickle.load(f)

    # Function to create nested defaultdicts
    def _callable():
        return defaultdict(set)

    logger.info('Processing WikiData relations...')
    wikidata = defaultdict(_callable)
    for subject, relations in relation_db.items():
        for relation, object in relations:
            if object['type'] != 'wikibase-entityid':
                continue
            object = object['value']['id']
            wikidata[relation][subject].add(object)

    logger.info('Processing annotated relations...')
    annotated_data = defaultdict(_callable)
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            for annotation in data['annotations']:
                subject = annotation['id']
                relations = annotation['relation']
                objects = annotation['parent_id']
                for relation, object in zip(relations, objects):
                    annotated_data[relation][subject].add(object)

    logger.info('Writing outputs...')
    with open(FLAGS.prefix + '.wikidata.csv', 'w') as f:
        writer = csv.writer(f)
        for relation, edges in wikidata.items():
            writer.writerow([readable(relation, alias_db), *fan_stats(edges)])

    with open(FLAGS.prefix + '.annotated_data.csv', 'w') as f:
        writer = csv.writer(f)
        for relation, edges in annotated_data.items():
            writer.writerow([readable(relation, alias_db), *fan_stats(edges)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('prefix', type=str)
    parser.add_argument('--alias_db', type=str, default='data/alias.pkl')
    parser.add_argument('--relation_db', type=str, default='data/relation.pkl')
    FLAGS, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    main(_)

