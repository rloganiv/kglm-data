"""
Utilities.
"""
from typing import Any, Dict, Generator, List, Set, Optional, Tuple
import gzip
import json
import logging
import bz2
from xml.etree import ElementTree
import re

from tqdm import tqdm
from spacy.tokens.doc import Doc
from spacy.parts_of_speech import NUM, SPACE

logger = logging.getLogger(__name__)


LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
RE_ENTITY = re.compile(r'Q\d+')
xmlns = '{http://www.mediawiki.org/xml/export-0.10/}'

def generate_instances(fname: str) -> Generator[Dict[str, Any], None, None]:
    """Generates instances from a JSON-lines file"""
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


def flatten_tokens(tokens: List[List[str]]) -> List[str]:
    return [word for sent in tokens for word in sent]


def format_wikilink(wikilink: str) -> str:
    """Formats a wikilink"""
    wikilink = wikilink.replace(' ', '_')
    if len(wikilink) == 1:
        return wikilink.capitalize()
    else:
        return wikilink[0].capitalize() + wikilink[1:]


def generate_from_wikidump(fname: str) -> Generator[Dict[str, Any], None, None]:
    """Generates data from a wikidata dump"""
    with gzip.open(fname) as f:
        for line in tqdm(f):
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


def wikipediadump_to_jsonl(fname: str) -> Generator[Dict[str, Any], None, None]:
    """Generate data from wikipedia dump"""

    def strip_tag_name(t):
        idx = t.rfind("}")
        if idx != -1:
            t = t[idx + 1:]
        return t

    def extract_title(element: ElementTree.Element) -> str:
        t = element.find(f'{xmlns}title')
        if t is None or t.text is None:
            logger.debug(f'{element} has no <title> element')
            return
        return t.text.replace(' ', '_').capitalize()

    with bz2.open(fname, 'r') as f:
        for event, elem in ElementTree.iterparse(f, events=("start", "end")):
            root = None
            if event == "start":
                if root is None:
                    root = elem
                else:
                    continue
                tname = strip_tag_name(elem.tag)
                if tname == "page":
                    title = extract_title(elem)
                    redirect = extract_redirect(elem)
                    out = {}
                    if title is not None:
                        out['title'] = title
                    if redirect is not None:
                        out['redirect_to'] = redirect[1]
                    elem.clear()
                    root.clear()
                    yield out


def extract_redirect(elem: ElementTree.Element) -> Optional[Tuple[str, str]]:
    """Extracts redirects from a <page> element in the Wikipedia dump.

    Args:
        elem : ``xml.etree.ElementTree.Element``
            The <page> element to process.

    Returns:
        A tuple ``(from, to)`` containing the titles of the pages being
        redirected from and to if the page is a redirect, otherwise ``None``.
    """
    # Get page title
    title = elem.find(f'{xmlns}title')
    if title is None or title.text is None:
        logger.debug('<page> has no <title> element')
        return
    _from = title.text.replace(' ', '_').capitalize()
    # Check if page is a redirect
    redirect = elem.find(f'{xmlns}redirect')
    if redirect is None:
        logger.debug('<page> has no <redirect> element')
        return
    _to = redirect.attrib['title'].replace(' ', '_').capitalize()
    logger.debug('Redirect from "%s" to "%s"', _from, _to)
    return _from, _to

def merge_numbers_to_single_token(doc: Doc) -> None:
    with doc.retokenize() as retokenizer:
        is_number = False
        start = 0
        end = 0
        for t in doc:
            if t.pos == NUM:
                if is_number:
                    end = t.i
                else:
                    start = t.i
                    end = t.i
                    is_number = True
            elif t.pos == SPACE and is_number:
                end = t.i
            else:
                if is_number:
                    if end > start:
                        retokenizer.merge(doc[start:end + 1])
                    is_number = False
        if is_number and end > start:
            retokenizer.merge(doc[start:end + 1])
    return doc
