"""
Filters enwiki dump to only 'Good' and 'Featured' articles
"""
import argparse
import bz2
import json
import logging
import re

from src.wikidump_to_jsonl import page_generator

logger = logging.getLogger(__name__) # pylint: disable=invalid-name
RE_GF = re.compile(r'{{([Gg]ood|[Ff]eatured) [Aa]rticle}}')


def main(_):
    for title, wikitext in page_generator(FLAGS.input):
        if RE_GF.search(wikitext):
            out = {
                'title': title,
                'wikitext': wikitext
            }
            print(json.dumps(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    FLAGS, _ = parser.parse_known_args()

    main(_)

