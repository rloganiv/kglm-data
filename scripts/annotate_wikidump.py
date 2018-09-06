#! /usr/bin/env python3.6
"""
Parse a Wikipedia dump.
"""

import argparse
import bz2
import csv
from collections import defaultdict, deque, namedtuple
import logging
import re
import sys
from typing import List, Tuple
from xml.etree import ElementTree

from mwparserfromhell import parse
from mwparserfromhell.nodes import Tag, Template, Text, Wikilink
import spacy
from sqlitedict import SqliteDict
from titlecase import titlecase

from moses import MosesTokenizer
from render import process_literal


logger = logging.getLogger(__name__)

logger.info('Loading SpaCy coreference model')
nlp = spacy.load('en_coref_lg')
spacy.tokens.Token.set_extension('wikidata_id', default=None)


xmlns = '{http://www.mediawiki.org/xml/export-0.10/}'
re_comment = re.compile(r'<!--.*?-->')
re_table_or_list = re.compile(r'(^\{\||^\||\*|!).*?$\s+', re.MULTILINE)
re_image = re.compile(r'(File|Image|ALTERNATIVE_ALIASES|Category):', re.IGNORECASE)
re_ref = re.compile(r'<ref([^<]*?\/>|.*?</ref>)', re.MULTILINE | re.DOTALL)
re_li = re.compile(r'<li([^<]*?\/>|.*?</li>)', re.MULTILINE | re.DOTALL)
irrelevant_templates = ['see also', 'main']
tokenizer = MosesTokenizer()


EntityToken = namedtuple('EntityToken', ['text', 'id'])


def link_by_coref(tokens):
    # Create a doc using predefined tokens (e.g. do not run SpaCy's tokenizer)
    text = [token.text for token in tokens]
    ids = [token.id for token in tokens]
    doc = spacy.tokens.Doc(nlp.vocab, words=text)
    # Annotate tokens
    for token, id in zip(doc, ids):
        token._.wikidata_id = id
    # Apply rest of nlp pipeline to get coreference annotations
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    # For each coreference cluster check if there is a unique wikidata id. If
    # there is assign it to each token in the cluster.
    if doc._.coref_clusters is None:  # All that work for nothing!
        return tokens
    for cluster in doc._.coref_clusters:
        id_set = set()
        for span in cluster:
            for token in span:
                if token._.wikidata_id is not None:
                    id_set.add(token._.wikidata_id)
        if len(id_set) == 1:
            id = id_set.pop()
            for span in cluster:
                for token in span:
                    token._.wikidata_id = id
    # Return the processed list of tokens
    out = [EntityToken(token.text, token._.wikidata_id) for token in doc]
    return out



class EnhancedToken(object):
    """A token storing all of the neccessary data to train REALM.

    Args:
        text : ``str``
            The text displayed when token is rendered.
        z : ``bool``
            Whether or not the current token is generated from the KG.
        parent : ``str``
            The entity id of the parent of the current token.
        relation : ``str``
            The selected relation used to pick the entity/fact to render.
        child : ``str``
        If an entity is selected, then the entity's id. Otherwise, the
        canonical representation of the fact literal.
    """
    def __init__(self,
                 text: str,
                 z: bool = False,
                 parent: str = None,
                 relation: str = None,
                 child: str = None,
                 type: str = None) -> None:
        if z:
            assert parent is not None
            assert relation is not None
            assert child is not None
        self.text = text
        self.z = z
        self.parent = parent
        self.relation = relation
        self.child = child
        self.type = type

    def __repr__(self):
        args = (self.text, self.z, self.parent, self.relation, self.child,
                self.type)
        return 'EnhancedToken(text="%s", z=%s, parent="%s",'\
               'relation="%s", child="%s", type="%s")' % args


# TODO: Define literal rendering functions
class RealmLinker(object):
    """Links sequences of tokens in Wikipedia articles to Wikidata
    entities/literals.

    Args:
        alias_db : ``sqlitedict.SqliteDict``
            Key-value store mapping an entity's id to a list of aliases for the
            entity.
        relation_db : ``sqlitedict.SqliteDict``
            Key-value store mapping an entity's id to a list of related
            entities.
        wiki_db : ``sqlitedict.SqliteDict``
            Key-value store mapping wikipedia titles to entity ids.
        max_len : ``int``
            Maximum possible length of linked token subsequence.
    """
    def __init__(self,
                 alias_db: SqliteDict,
                 relation_db: SqliteDict,
                 wiki_db: SqliteDict,
                 max_len: int = 10):
        self.alias_db = alias_db
        self.wiki_db = wiki_db
        self.relation_db = relation_db
        self.max_len = max_len
        self.seen = None  # Which entities have been observed
        self.parents = None  # Link 1st step entities back to parents
        self.reverse_aliases = None  # Alias -> EntityId lookup
        self.prior = None  # Entities that were not in KG when linked (no parents)

    def instantiate(self, title: str) -> None:
        """Instantiates the ``RealmLinker``. This function should be called
        before linking a new document.

        Args:
            title : ``str``
                Title of the wikipedia page being linked.
        """
        # Clear information
        self.seen = set()
        self.parents = dict()
        self.reverse_aliases = dict()
        self.prior = []

        # Instantiate with top-level entity + relations
        entity_id = self.wiki_db[title]
        self.expand(entity_id)

    def expand(self, id):
        """Adds first order relations out from entity ``id`` to the graph."""
        try:
            aliases = self.alias_db[id]
        except KeyError:  # Object is probably a literal and cannot be expanded
            return
        for alias in aliases:
            alias_tokens = tuple(tokenizer.tokenize(alias))
            self.reverse_aliases[alias_tokens] = id
        relations = self.relation_db[id]
        for prop, value in relations:
            # If value is an entity then look up in db
            if value['type'] == 'wikibase-entityid':
                child_id = value['value']['id']
                try:
                    child_aliases = self.alias_db[child_id]
                except:  # TODO: What does this mean???
                    return
            # Otherwise it is a literal that must be processed
            else:
                child_id, child_aliases = process_literal(value)
            # Skip if no identifier was extracted (e.g. if literal is a URL)
            if child_id is None:
                continue
            # Update parents
            self.parents[child_id] = (prop, id)
            # Add all of the aliases to the reverse lookup
            for child_alias in child_aliases:
                child_alias_tokens = tuple(tokenizer.tokenize(child_alias))
                if child_alias_tokens in self.reverse_aliases:
                    # Oldest alias recieves preference
                    # logger.warning('Collision, alias="%s"', child_alias_tokens)
                    pass
                else:
                    self.reverse_aliases[child_alias_tokens] = child_id
        self.seen.add(id)

    def link(self, tokens: List[Tuple[str]]) -> List[str]:
        """Attempts to link a sequence of tokens.

        Args:
            tokens : List[Tuple[str]]

        Returns:
        """
        out = []
        token_stack = deque(reversed(tokens))
        while len(token_stack) > 0:
            logging.debug('Token stack length: %i', len(token_stack))
            active = token_stack.pop()
            tmp_stack = deque()
            tmp_stack.append(active)
            id = None
            if active.id is None:  # Need to try and match to an alias
                for _ in range(self.max_len):
                    try:
                        tmp = token_stack.pop()
                    except IndexError:  # Trying to pop from empty stack
                        break
                    if tmp.id is not None:
                        token_stack.append(tmp)
                        break
                    else:
                        tmp_stack.append(tmp)
                # See if current subsequence is an alias. If it is then find
                # the id and proceed, otherwise keep popping
                while len(tmp_stack) > 0:
                    alias = tuple(x.text.lower() for x in tmp_stack)
                    if alias in self.reverse_aliases:
                        id = self.reverse_aliases[alias]
                        break
                    else:
                        tmp = tmp_stack.pop()
                        if len(tmp_stack) > 0:
                            token_stack.append(tmp)
            else:  # Found already linked token
                while True:
                    tmp = token_stack.pop()
                    if tmp.id != active.id:
                        token_stack.append(tmp)
                        break
                    else:
                        tmp_stack.append(tmp)
                id = active.id
            # If nothing has been linked, then token is plain text.
            if id is None:
                out.append(EnhancedToken(active.text))
            # Otherwise everything in ``tmp_stack`` is part of the alias. To
            # complete the annotation we need to find a path back to a seen
            # entity.
            else:
                # If it has been seen before or doesn't have parents then the
                # relation that generates it is reflexive.
                if id in self.seen or id not in self.parents:
                    relation = ('@@REFLEXIVE@@', None)
                    parent = id
                # Otherwise it hasn't been seen but is linked to something that
                # has.
                else:
                    relation, parent = self.parents[id]
                self.expand(id)  # Expand no matter what
                while len(tmp_stack) > 0:
                    tmp = tmp_stack.popleft()
                    token = EnhancedToken(text=tmp.text, z=True, parent=parent,
                                          relation=relation, child=id,
                                          type=None)  # TODO: Fix type
                    out.append(token)
        return out


def process_tag(node: Tag):
    while isinstance(node, Tag):
        if str(node.closing_tag) in ['b', 'i']:
            node = node.contents
        elif str(node.closing_tag) == 'math':
            return [Text('@MATH@')]
        else:
            return None
    return node.filter(recursive=False)


def render_wikilink(node: Wikilink):
    if re_image.match(str(node.title)):
        return
    if node.text is not None:
        return str(node.text)
    else:
        return str(node.title)


def link_entity(node: Wikilink,
                wiki_db: SqliteDict) -> str:
    """Links a ``Wikilink`` node to a Wikidata entity.

    Args:
        node : ``mwparserfromhell.node.Wikilink``
            The wikilink to link.
        wikidb : ``sqlitedict.SqliteDict``
            Maps enwiki page titles to entities.

    Returns:
        A ``str`` containing the corresponding entity id if found, otherwise
        None.
    """
    key = str(node.title)
    key = key.lower().replace(' ', '_')
    logger.debug('Looking up "%s" in wikidb', key)
    try:
        entity_id = wiki_db[key]
        logger.debug('Found entity id "%s"', entity_id)
    except KeyError:
        entity_id = None
        logger.debug('Could not find "%s" in wikidb', key)
    # if entity_id is None:
    #     alternate_key = titlecase(key)
    #     try:
    #         logger.debug('Attempting to use alternate key "%s"', alternate_key)
    #         entity_id = wiki_db[alternate_key]
    #         logger.debug('Found entity id "%s"', entity_id)
    #     except KeyError:
    #         logger.debug('Still no luck')
    return entity_id


def clean_wikitext(wikitext: str) -> str:
    """Removes tables and lists from wikitext.

    Args:
        wikitext : ``str``
            The raw wikitext.

    Returns:
        A ``str`` containing the processed wikitext.
    """
    out = re_comment.sub('', wikitext)
    out = re_ref.sub('', out)
    out = re_li.sub('', out)
    # out = re_title.sub('', out)
    out = re_table_or_list.sub('', out)
    return out


def process_wikitext(wikitext, wiki_db, title):
    """Processes wikitext.

    This entails parsing the text, linking Wikilinks to corresponding Wikidata
    entities, tokenizing the text, ...

    Args:
        text : ``str``
            The string containing the wikitext.

    Returns:
        TODO: FIGURE THIS OUT
    """
    wikicode = parse(wikitext)

    # TODO: Hop and grab aliases of associated entities
    tokens = []
    nodes = deque(wikicode.filter(recursive=False))
    while len(nodes) > 0:
        node = nodes.popleft()
        entity_id = None
        text = None
        # A tag may or may not contain valid tokens, if it does insert the
        # inner nodes to the front of the queue.
        if isinstance(node, Tag):
            processed_tag = process_tag(node)
            if processed_tag is not None:
                nodes.extendleft(reversed(processed_tag))
        # Text can simply be rendered as a string
        elif isinstance(node, Text):
            text = str(node)
        # Wikilinks should be rendered as a string as well as possibly linked
        # to an entity in the knowledge base.
        elif isinstance(node, Wikilink):
            text = render_wikilink(node)
            entity_id = link_entity(node, wiki_db)
        elif isinstance(node, Template):
            if str(node.name) == 'math':
                tokens.append(('@MATH@', None))
            if str(node.name) not in irrelevant_templates:
                try:
                    if tokens[-1][0] != '@UNK-TEMPLATE@':
                        tokens.append(EntityToken('@UNK-TEMPLATE@', None))
                except IndexError:
                    pass
        if text is None:
            continue
        # TODO: Check if text is a literal, and token[1] the canonical
        # representation (e.g. date -> POSIX format).
        tokens.extend(EntityToken(x, entity_id) for x in tokenizer.tokenize(text))
    # Remove trailing @UNK-TEMPLATE@ token if it exists
    if tokens[0][0] == '@UNK-TEMPLATE@':
        tokens = tokens[1:]
    if tokens[-1][0] == '@UNK-TEMPLATE@':
        tokens = tokens[:-1]
    return tokens


def process_page(elem: ElementTree.Element) -> str:
    """Extracts title and wikitext from a <page> element in the Wikipedia dump.

    Args:
        elem : ``xml.etree.ElementTree.Element``
            The <page> element to process.

    Returns:
        A ``str`` containing the page's wikitext if found, otherwise ``None``.
    """
    # Get page title
    title = elem.find(f'{xmlns}title')
    if title is None:
        logger.debug('<page> has no <title> element')
        return
    logger.debug('<title>: %s', title.text)
    # Check if page is a redirect
    if elem.find(f'{xmlns}redirect') is not None:
        logger.debug('<page> is <redirect>')
        return
    # Get revision element (which contains the wikitext)
    revision = elem.find(f'{xmlns}revision')
    if revision is None:
        logger.debug('<page> has no <revision> element')
        return
    # Ensure that the text is wikitext formatted
    model = revision.find(f'{xmlns}model')
    if model is None:
        logger.debug('<revision> has no <model> element')
        return
    if model.text != 'wikitext':
        logger.debug('<model> is not "wikitext"')
        return
    # Extract text
    text = revision.find(f'{xmlns}text')
    if text is None:
        logger.debug('<revision> has no <text> element')
        return
    return title.text, text.text


def main(_):
    alias_db = SqliteDict(FLAGS.alias_db)
    relation_db = SqliteDict(FLAGS.relation_db)
    wiki_db = SqliteDict(FLAGS.wiki_db)
    linker = RealmLinker(alias_db, relation_db, wiki_db)
    fieldnames = ['page', 'line', 'text', 'z', 'parent', 'relation', 'child',
                  'type']
    writer = csv.DictWriter(sys.stdout, delimiter='\t', fieldnames=fieldnames)
    with bz2.open(FLAGS.input, 'r') as f:
        tree = ElementTree.iterparse(f, events=('start', 'end'))
        root = None
        page_count = 0
        for event, elem in tree:
            # Need to catch root so we can clear the parse tree (otherwise
            # we'll run out of memory).
            if event == 'start':
                if root is None:
                    root = elem
                    logger.debug('ROOT: %s', root)
                else:
                    continue
            if elem.tag == f'{xmlns}page':
                processed = process_page(elem)
                if processed is None:
                    continue
                title, wikitext = processed
                title = title.lower().replace(' ', '_')
                linker.instantiate(title)
                wikitext = clean_wikitext(wikitext)
                tokens = process_wikitext(wikitext, wiki_db, title)
                tokens = link_by_coref(tokens)
                tokens = linker.link(tokens)
                # Serialization
                for i, token in enumerate(tokens):
                    d = {'page': page_count, 'line': i, **token.__dict__}
                    writer.writerow(d)
                elem.clear()
                root.clear()
                page_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--alias_db', type=str, default='data/alias.db')
    parser.add_argument('--relation_db', type=str, default='data/relation.db')
    parser.add_argument('--wiki_db', type=str, default='data/wiki.db')
    parser.add_argument('--debug', action='store_true')
    FLAGS, _ = parser.parse_known_args()
    if FLAGS.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    main(_)

