"""
Split train/test/valid sets of KGLM data that match Wikitext-103
"""
import argparse
import json
import logging

logger = logging.getLogger(__name__)


def main(_):

    with open(FLAGS.valid, 'r') as f:
        valid_set = set(x.strip() for x in f)
    logger.info('Loaded %i validation titles from "%s"',
                len(valid_set),
                FLAGS.valid)

    with open(FLAGS.test, 'r') as f:
        test_set = set(x.strip() for x in f)
    logger.info('Loaded %i test titles from "%s"',
                len(test_set),
                FLAGS.test)

    with open(FLAGS.mini, 'r') as f:
        mini_set = set(x.strip() for x in f)
    logger.info('Loaded %i titles from "%s"',
                len(mini_set),
                FLAGS.mini)

    logger.info('Splitting data in "%s"', FLAGS.input)
    train_file = open(FLAGS.prefix + '.train.jsonl', 'w')
    valid_file = open(FLAGS.prefix + '.valid.jsonl', 'w')
    test_file = open(FLAGS.prefix + '.test.jsonl', 'w')
    mini_file = open(FLAGS.prefix + '.mini.jsonl', 'w')
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            title = data['title']
            if title in valid_set:
                valid_file.write(line)
                valid_set.remove(title)
            elif title in test_set:
                test_file.write(line)
                test_set.remove(title)
            else:
                train_file.write(line)
            if title in mini_set:
                mini_file.write(line)
                mini_set.remove(title)

    logger.info('Splits completed')
    logger.info('Remaining validation titles: %s', valid_set)
    logger.info('Remaining test titles: %s', test_set)
    logger.info('Remaining mini titles: %s', mini_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('prefix', type=str)
    parser.add_argument('-v,', '--valid', type=str, required=True)
    parser.add_argument('-t', '--test', type=str, required=True)
    parser.add_argument('-m', '--mini', type=str, required=True)
    FLAGS, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    main(_)

