"""
Utilities.
"""
from typing import Any, Dict, Generator, Set

import gzip
import json
import logging

logger = logging.getLogger(__name__)


LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

def format_wikilink(wikilink: str) -> str:
    """Formats a wikilink"""
    wikilink = wikilink.replace(' ', '_')
    if len(wikilink) == 1:
        return wikilink.capitalize()
    else:
        return wikilink[0].capitalize() + wikilink[1:]
    return wikilink


def generate_from_wikidump(fname: str) -> Generator[Dict[str, Any], None, None]:
    """Generates data from a wikidata dump"""
    with gzip.open(fname) as f:
        for i, line in enumerate(f):
            if not i % 1000:
                logger.info('On line %i', i)
            if line[0] == '[':
                line = line[1:]
            elif line[-1] == ']':
                line = line[:-1]
            else:
                line = line[:-2]
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning('Could not decode line to JSON:\n"%s"\n', line)
                continue
            yield data


def load_allowed_entities(fname: str) -> Set[str]:
    """Loads a set of allowed entities from a txt file."""
    if fname is None:
        logger.info('Entities not restricted')
        return
    else:
        logger.info('Loading allowed entities from: "%s"', fname)
        allowed_entities = set()
        with open(fname, 'r') as f:
            for line in f:
                allowed_entities.add(line.strip())
        logger.info('%i allowed entities found', len(allowed_entities))
        return allowed_entities

