import argparse
from collections import Counter, defaultdict
import json
import logging
import pickle
import re
from statistics import mean

logger = logging.getLogger(__name__)


MANUAL_DELETIONS = [
    re.compile('P.*:P585'),  # Point in time qualifier
    re.compile('P530')  # Diplomatic relation
]

def _callable():
    return defaultdict(set)


def main(_):

    # Load the relation database
    logger.info('===Loading relation db===')
    with open(FLAGS.relation_db, 'rb') as f:
        relation_db = pickle.load(f)

    # Store sets of all of the possible tail entities for a given
    # subject-relation. To facilitate global vs. per entity pruning we order
    # keys both ways.
    logger.info('===Building edge sets===')
    rs = defaultdict(_callable)
    sr = defaultdict(_callable)
    for subject, relations in relation_db.items():
        for relation, object in relations:
            if object['type'] != 'wikibase-entityid':
                continue
            object = object['value']['id']
            rs[relation][subject].add(object)
            sr[subject][relation].add(object)

    # Identify relations which introduce (on average) more than ``max_avg_fan``
    # nodes to the KG when expanded.
    logger.info('===Detecting global bad relations===')
    global_bad_relations = set()
    for relation, edges in rs.items():
        avg = mean(len(x) for x in edges.values())
        if avg >= FLAGS.max_avg_fan:
            logger.info('\t%s', relation)
            global_bad_relations.add(relation)

    # Identify subject-relation pairs which introduce more than ``max_child``
    # nodes to the KG when expanded.
    # logger.info('===Detecting per-entity bad relations===')
    # per_entity_bad_relations = dict()
    # for subject, edges in sr.items():
    #     bad_relations = set(x for x, y in edges.items()
    #                         if len(y) > FLAGS.max_avg_fan
    #                         and x not in global_bad_relations)
    #     per_entity_bad_relations[subject] = bad_relations
    #     for relation in bad_relations:
    #         logger.info('\t%s\t%s', subject, relation)

    # Obtain observed counts of relations from the annotated training data.
    logger.info('===Obtaining relation counts===')
    relation_counts = Counter()
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
        for annotation in data['annotations']:
            for relation in annotation['relation']:
                relation_counts[relation] += 1
    logger.info('===Detecting low freq. bad relations===')
    low_freq_relations = set(x for x,y in relation_counts.items() if y <= FLAGS.min_count)
    for relation in low_freq_relations:
        logger.info('\t%s', relation)

    # Prune
    logger.info('===Pruning===')
    pruned_relation_db = dict()
    for entity, children in relation_db.items():
        pruned_children = []
        for relation, value in children:
            if any(regex.search(relation) for regex in MANUAL_DELETIONS):
                logger.info('Manual: %s in %s', relation, entity)
            elif relation in global_bad_relations:
                logger.info('Global: %s in %s', relation, entity)
            # elif entity in per_entity_bad_relations:
            #     if relation in per_entity_bad_relations[entity]:
            #         logger.info('Per-entity: %s in %s', relation, entity)
            elif relation in low_freq_relations:
                logger.info('Low-freq: %s in %s', relation, entity)
            else:
                logger.info('Survived: %s in %s', relation, entity)
                pruned_children.append((relation, value))
        pruned_relation_db[entity] = pruned_children
    with open(FLAGS.output, 'wb') as f:
        pickle.dump(pruned_relation_db, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--output', type=str, default='data/relation.pruned.pkl')
    parser.add_argument('--relation_db', type=str, default='data/relation.pkl')
    parser.add_argument('--max_avg_fan', type=int, default=15,
                        help='Relations whose average fan out are >= this '
                             'number are pruned')
    parser.add_argument('--max_child', type=int, default=100,
                        help='Relation edges for a given entity which are >= '
                             'this number are pruned')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimum number of times a relation must appear '
                             'in the dataset in order to be kept')
    FLAGS, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    main(_)

