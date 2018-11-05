"""
Submits requests to the enwiki parse API.
"""
import argparse
import json
import logging
import requests

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main(_):
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
    logger.info('Endpoint: %s', endpoint)
    logger.info('Headers: %s', headers)
    with open(FLAGS.input, 'r') as f:
        for i, line in enumerate(f):
            if not i % 100:
                logger.info('On line %i', i)
            data = json.loads(line)
            params = base_params.copy()
            params['page'] = data['title']
            logger.debug('Params: %s', params)

            response = requests.get(endpoint, params=params, headers=headers)
            logger.debug('Response: %s', response)

            if response.status_code:
                response_json = response.json()
                try:
                    html = response_json['parse']['text']['*']
                except KeyError:
                    html = None
                    logger.warning('No HTML returned for "%s"', data['title'])
                out = {
                    'title': data['title'],
                    'html': html
                }
                print(json.dumps(out))
            else:
                logger.warning('Bad response for "%s"', data['title'])
            del params
    logger.info('Processed %i lines', i+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL)

    main(_)

