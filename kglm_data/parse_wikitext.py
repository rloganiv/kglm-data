"""
Submits requests to the enwiki parse API.
"""
import argparse
import json
import logging
import requests
import multiprocessing as mp
from typing import Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
print_lock = mp.Lock()


def get_wiki_page(title: str, lang: str) -> Optional[str]:
    if title.endswith("?"):
        title = title[:-1]
    elif title.endswith("%3F"):
        title = title[:-3]
    endpoint = f"https://{lang}.wikipedia.org/wiki/{title}?action=render"

    response = requests.get(endpoint)

    if response.status_code:
        return response.content.decode("utf-8")
    else:
        logger.warning(f'Bad response for {title}: {response}')
        return None


def get_en_wiki_page(title: str) -> Optional[str]:
    endpoint = 'https://en.wikipedia.org/w/api.php'
    headers = {
        'User-Agent': 'Robert L. Logan IV (UC Irvine)',
        'From': 'rlogan@uci.edu'
    }
    base_params = {
        'action': 'parse',
        'prop': 'text',
        'format': 'json',
        'contentmodel': 'wikitext'
    }
    params = base_params.copy()
    params['page'] = title
    logger.debug('Params: %s', params)

    response = requests.get(endpoint, params=params, headers=headers)
    logger.debug('Response: %s', response)

    if response.status_code:
        response_json = response.json()
        try:
            html = response_json['parse']['text']['*']
        except KeyError:
            html = None
            logger.warning('No HTML returned for "%s"', title)

        return html
    else:
        logger.warning('Bad response for "%s"', title)
        return None


def loader(q: mp.JoinableQueue, FLAGS: Tuple[Any]):
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


def worker(q: mp.JoinableQueue, output, FLAGS: Tuple[Any]) -> None:
    """Retrieves items from the queue and enriches them by fetching the HTML of the page"""
    while True:
        logger.debug(f'{mp.current_process().name} taking a task from the queue')
        json_data = q.get()
        if json_data is None:
            break
        try:
            if FLAGS.language == "en":
                html = get_en_wiki_page(json_data['title'])
            else:
                html = get_wiki_page(json_data['title'], FLAGS.language)
        except:
            html = None
            logger.warning(f'Error fetching HTML for {json_data["title"]}')
        out = {
            'title': json_data['title'],
            'html': html
        }
        print_lock.acquire()
        output.write(json.dumps(out)+'\n')
        print_lock.release()
        q.task_done()
        logger.debug(f'{mp.current_process().name} finished a task')


def main(_):
    logger.info('Starting queue loader')
    work_queue = mp.JoinableQueue(maxsize=256)
    loader_process = mp.Process(target=loader, args=(work_queue, FLAGS))
    loader_process.start()

    logger.info('Launching worker processes')
    output = open(FLAGS.output, 'w')
    processes = [mp.Process(target=worker, args=(work_queue, output, FLAGS)) for _ in range(FLAGS.j)]
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
    parser = argparse.ArgumentParser("Add HTML to Wikipedia dump to be able to parse links later on. "
                                     "Requires .jsonl file with 'title' for Wikipedia pages as input")
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('-j', type=int, default=1,
                        help='Number of processors')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--language', type=str, default='en')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL)

    main(_)

