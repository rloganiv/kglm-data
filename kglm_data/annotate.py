# pylint: disable=redefined-builtin,redefined-outer-name
"""
Adds entity annotations to JSON-lines files.
"""
from typing import Any, Tuple
import argparse
import json
import logging
from multiprocessing import JoinableQueue, Lock, Process, current_process
import pickle
from time import time

from sqlitedict import SqliteDict

from kglm_data.annotator import Annotator
from kglm_data.util import LOG_FORMAT

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
print_lock = Lock()


def worker(q: JoinableQueue, output, FLAGS: Tuple[Any]) -> None:
    """Retrieves files from the queue and annotates them."""
    if FLAGS.in_memory:
        with open(FLAGS.alias_db, 'rb') as f:
            alias_db = pickle.load(f)
        with open(FLAGS.relation_db, 'rb') as f:
            relation_db = pickle.load(f)
        with open(FLAGS.wiki_db, 'rb') as f:
            wiki_db = pickle.load(f)
    else:
        alias_db = SqliteDict(FLAGS.alias_db, flag='r')
        relation_db = SqliteDict(FLAGS.relation_db, flag='r')
        wiki_db = SqliteDict(FLAGS.wiki_db, flag='r')

    annotator = Annotator(alias_db,
                          relation_db,
                          wiki_db,
                          distance_cutoff=FLAGS.cutoff,
                          match_aliases=FLAGS.match_aliases,
                          unmatch=FLAGS.unmatch,
                          prune_clusters=FLAGS.prune_clusters,
                          language=FLAGS.language,
                          merge_entities=FLAGS.merge_entity_tokens,
                          spacy_model_path=FLAGS.spacy_model_path)
    while True:
        t0 = time()
        logger.debug(f'{current_process().name} taking a task from the queue')
        json_data = q.get()
        if json_data is None:
            break
        annotation = annotator.annotate(json_data)
        print_lock.acquire()
        output.write(json.dumps(annotation)+'\n')
        print_lock.release()
        q.task_done()
        t1 = time()
        logger.debug(f'{current_process().name} finished {json_data["title"]} in {t1-t0} seconds')


def loader(q: JoinableQueue, FLAGS: Tuple[Any]) -> None:
    i = 0
    with open(FLAGS.input, 'r') as f:
        for i, line in enumerate(f):
            if line == '{}\n':
                continue
            try:
                q.put(json.loads(line.strip()))
            except:
                logger.warning(f"Unable to load {line} into queue")
    logger.info('All %i lines have been loaded into the queue', i+1)


def main(_):  # pylint: disable=missing-docstring
    if FLAGS.prune_clusters:
        logger.warning('Cluster pruning is active. Beware the doc._.clusters is '
                       'not updated to only hold pruned mentions - this may '
                       'cause unexpected behavior if ``Annotator._propagate_ids()`` '
                       'is called multiple times.')
    logger.info('Starting queue loader')
    work_queue = JoinableQueue(maxsize=256)
    loader_process = Process(target=loader, args=(work_queue, FLAGS))
    loader_process.start()

    logger.info('Launching worker processes')
    output = open(FLAGS.output, 'w')
    processes = [Process(target=worker, args=(work_queue, output, FLAGS)) for _ in range(FLAGS.j)]
    for p in processes:
        p.start()

    loader_process.join()
    work_queue.join()
    for _ in range(FLAGS.j):
        work_queue.put(None)
    for p in processes:
        p.join()
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-j', type=int, default=1,
                        help='Number of processors')
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    parser.add_argument('--relation_db', type=str, default='data/relation.db')
    parser.add_argument('--wiki_db', type=str, default='data/wiki.db')
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--cutoff', '-c', type=float, default=float('inf'),
                        help='Maximum distance between related entities')
    parser.add_argument('--match_aliases', '-m', action='store_true',
                        help='Whether or not to exact match aliases of entities '
                             'whose wikipedia links appear in the document')
    parser.add_argument('--unmatch', '-u', action='store_true',
                        help='Whether or not to ignore ids from NEL which do '
                             'not match ids from wikipedia')
    parser.add_argument('--prune_clusters', '-p', action='store_true',
                        help='Whether or not to prune mentions from clusters '
                             'that are not aliases or pronouns')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--language', type=str, default="en")
    parser.add_argument('--merge_entity_tokens', action='store_true',
                        help='Whether or not to merge SpaCy NER token spans into single '
                             'token (can help expand spans for coreference and KG annotations)')
    parser.add_argument('--spacy_model_path', type=str, default=None,
                        help='Override language (for Spacy model) by loading model from local path')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=LEVEL)

    main(_)
