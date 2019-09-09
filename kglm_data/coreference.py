import json
import logging
import re
from collections import Counter
from typing import List, Dict, Any, Callable, Tuple, Iterable, Union, Optional
from itertools import takewhile

import spacy
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token
import gender_guesser.detector as gender

from kglm_data.util import flatten_tokens

# ----------------
# Helper functions
# ----------------


def _merge_clusters(clusters):
    def _tuplify_set(x):
        if not isinstance(x, set):
            return x
        else:
            return tuple(_tuplify_set(elt) for elt in sorted(x, key=lambda tpl: tpl[0]))

    def _min_span(cluster):
        return min(x[0] for x in cluster)

    mapping = dict()
    for cluster in clusters:
        cluster = set(tuple(x) for x in cluster)
        # Merge existing clusters into current cluster
        for span in cluster.copy():
            if span in mapping:
                cluster.update(mapping[span])
        # Update all spans to point at current cluster
        for span in cluster:
            mapping[span] = cluster
    # The merged clusters are the unique values in ``mapping``. In order to use
    # the ``set()`` we need the clusters to be hashable, so we turn them into
    # tuples using ``_tuplify_set``.
    unique_clusters = list(set(_tuplify_set(x) for x in mapping.values()))
    unique_clusters = sorted(unique_clusters, key=_min_span)
    return list(unique_clusters)


def _token_to_span(token: Token):
    return Span(doc=token.doc,
                start=token.i,
                end=token.i + 1,
                vector=token.vector)


class CoreferenceResolver:
    """
    Class for custom coreference resolution.
    Has the same API as CoreNLPCorefPredictor found in realm_coref.py
    """
    pronoun_pos_tags_by_language = {'sv': {'PRON'}}
    pronoun_by_gender_by_language = {
        'sv': {
            'male': {'han', 'honom', 'hans', 'Han', 'Honom', 'Hans'},
            'female': {'hon', 'henne', 'hennes', 'Hon', 'Henne', 'Hennes'}
        }
    }
    genitive_re = re.compile('(:?s|\'|´)?\s*$')

    def __init__(self, language: str, spacy_model_dir: str) -> None:
        # TODO: Does it make sense to merge spacy NER spans into single tokens here?
        self.nlp: Language = spacy.load(spacy_model_dir, disable=['tokenizer', 'parser'])
        self.nlp.tokenizer = self.nlp.tokenizer.tokens_from_list
        self.pronoun_pos_tags = self.pronoun_pos_tags_by_language[language]
        self.pronouns_by_gender = self.pronoun_by_gender_by_language[language]
        self.all_pronouns = self.pronouns_by_gender['male'].union(self.pronouns_by_gender['female'])
        self.gender_detector = gender.Detector()
        self.logger = logging.getLogger(__name__)

    def is_pronoun(self, token: Token):
        # TODO: Do we want to tag possessive pronouns as well? If so, replace `== 'PRON'` -> `in ('PRON', 'DET')`
        return token.pos_ == 'PRON' and token.string.lower().strip() in self.all_pronouns

    @staticmethod
    def extract_propername_candidates(entities: List[Span]) -> Iterable[Tuple[Span, Iterable[Span]]]:
        return ((entities[i], reversed(entities[:i])) for i in range(len(entities) - 1))

    def extract_pronominal_candidates(self, entities: List[Span]) -> Iterable[Tuple[Span, Iterable[Span]]]:
        if not entities:
            return []

        def find_preceeding_entities(stop: int) -> Iterable[Span]:
            return reversed(list(takewhile(lambda e: e.start < stop, entities)))

        doc = entities[0].doc
        return ((_token_to_span(doc[i]), find_preceeding_entities(i)) for i in range(len(doc) - 1) if self.is_pronoun(doc[i]))

    def remove_gentitive(self, s: str) -> str:
        return self.genitive_re.sub('', s)

    def propername_match(self, ent: Span, antecedent: Span):
        """
        Does the entity (name) match the antecedent (name)?
        """
        ent_string = self.remove_gentitive(ent.string)
        antecedent_string = self.remove_gentitive(antecedent.string)

        if ent_string == antecedent_string:
            return True

        if ent.label_ in {'PRS', 'ORG'} and ent.label_ == antecedent.label_ and ent_string in antecedent_string:
            return True

        return False

    def pronominal_match(self, prn_span: Span, antecedent: Span):
        """
        Does the entity (pronoun) match the antecedent (name)?
        """
        prn = prn_span[0]
        if antecedent[0].ent_type_ == 'PRS':
            gender = self.get_gender(antecedent)
            if gender in {'male', 'female'} and prn.string.strip() in self.pronouns_by_gender[gender]:
                return True
        return False

    @staticmethod
    def merge_entities(doc: Doc, input_entities: List[List[Any]]):
        """
        Merge entities from Spacy's NER and the input wikidata entities.
        For a merged entity, we keep:
        - the spans of the wiki entity
        - the entity type of the spacy entity
        If no matching input entity is found, we keep the spacy entity as is
        """
        def create_merged_span(spacy_span: Span, wiki_entity: List[Any]):
            return Span(doc=spacy_span.doc,
                        start=wiki_entity[1],
                        end=wiki_entity[2],
                        vector=spacy_span.vector,
                        label=spacy_span.label)

        def spans_match(spacy_span: Span, wiki_entity: List[str]):
            return spacy_span.start == wiki_entity[1] or spacy_span.end == wiki_entity[2]

        def merge_or_get_doc_ent(spacy_span: Span):
            return next((create_merged_span(spacy_span, input_ent) for input_ent in input_entities if spans_match(spacy_span, input_ent)), spacy_span)

        def merge_or_keep_wiki_entity(wiki_entity: List[str]):
            return next((create_merged_span(spacy_span, wiki_entity) for spacy_span in doc.ents if spans_match(spacy_span, wiki_entity)), doc[wiki_entity[1]:wiki_entity[2]])

        spacy_spans = set([merge_or_get_doc_ent(doc_ent) for doc_ent in doc.ents])
        wiki_spans = set([merge_or_keep_wiki_entity(wiki_entity) for wiki_entity in input_entities])

        return sorted(list(spacy_spans.union(wiki_spans)), key=lambda x: x.start)

    def resolve_coreferences(self,
                             entities: List[Span],
                             lookback: int,
                             match: Callable[[Union[Span, Token], Union[Span, Token]], bool],
                             extract_candidates: Callable[[List[Span]], Iterable[Tuple[Span, Iterable[Span]]]]
                             ) -> List[Tuple[Span, Span]]:
        """
        Generalisation of the coreference resolution
        :param entities: List of (merged) entities
        :param lookback: Maximum token distance from entity to antecedent
        :param match: Function to decide if the two entities match
        :param extract_candidates: Function to extract valid antecedents as matching candidates
        :return: A list of tuples, where each tuple consists of (antecedent, entity)
        """

        def in_looback(ent, antecedent):
            def start(element):
                if isinstance(element, Span):
                    return element.start
                elif isinstance(element, Token):
                    return element.i
            return start(ent) > start(antecedent) and start(ent) - start(antecedent) < lookback

        def find_antecedent(ent: Span, antecedents: Iterable[Span]) -> Optional[Tuple[Span, Span]]:
            # TODO: Do we want to chain coreferences here? e.g. if an antecedent is a pronoun -> match
            return next(((antecedent, ent) for antecedent in antecedents
                         if in_looback(ent, antecedent) and match(ent, antecedent)), None)

        candidates = [find_antecedent(ent, antecedent) for ent, antecedent in extract_candidates(entities)]
        candidates = [coreference for coreference in candidates if coreference is not None]
        return candidates

    def predict_json(self, inputs: Dict[str, Any], print_summary=False, print_detailed=False) -> Dict[str, Any]:
        """
        Main method, named after the CoreNLPCorefPredictor found in realm_coref.py
        Takes an inputs dict, and returns the same dict augmented with the key "clusters" (corefererence clusters)
        """
        self.logger.debug(f"Flattening {inputs['title']}")
        tokens = flatten_tokens(inputs['tokens'])
        self.logger.info(f"Before flattening '{inputs['title']}': "
                         f"{len(inputs['tokens'])} tokens, after: {len(tokens)}")
        doc: Doc = self.nlp(tokens)

        entities = self.merge_entities(doc, inputs['entities'])
        propername_lookback = 500
        propername_clusters = self.resolve_coreferences(entities, propername_lookback, self.propername_match,
                                                        self.extract_propername_candidates)

        pronominal_lookback = 100

        pronominal_clusters = self.resolve_coreferences(entities, pronominal_lookback, self.pronominal_match,
                                                        self.extract_pronominal_candidates)

        inputs['clusters'] = _merge_clusters(
            [[[c[0].start, c[0].end - 1], [c[1].start, c[1].end - 1]] for c in
             propername_clusters + pronominal_clusters])
        if print_detailed:
            print(self.clusters_to_string(inputs['clusters'], doc)[0])
        if print_summary:
            print(self.clusters_to_string(inputs['clusters'], doc)[1])
        return inputs

    def get_gender(self, name: Span):
        """
        Guess the gender of a name, using https://pypi.org/project/gender-guesser/
        :param name:
        :return: one of "male", "female", "unknown" (and a couple of other genders)
        """
        gender = self.gender_detector.get_gender(name[0].string.strip())
        if gender.startswith('mostly_'):
            gender = gender.replace('mostly_', '')
        return gender

    def clusters_to_string(self, clusters, doc) -> Tuple[str, str]:
        """
        Print the clusters. Useful for debugging and creating slides for knowledge sharing ;)
        :param clusters:
        :param doc:
        :return: a tuple of strings, detailed and summary
        """
        detailed = ""
        summary = ""
        for idx, cluster in enumerate(clusters):
            detailed += f"\nCluster {idx}\n"
            mentions = Counter()
            for eidx, entity in enumerate(cluster):
                mentions[str(doc[entity[0]:entity[1] + 1])] += 1
                if doc[entity[0]].ent_type_:
                    ent_type = doc[entity[0]].ent_type_
                else:
                    ent_type = doc[entity[0]].pos_
                detailed += f"\t{eidx:4} {doc[entity[0]:entity[1]+1]} ({ent_type}) {entity[0]}:{entity[1]}\n"
            mentions_string = ', '.join([f"{token} ({count})" for token, count in mentions.items()])
            summary += f"{idx:3} ({sum(mentions.values())}): {mentions_string}\n"
        return detailed, summary


# Example of how to use the Coreference module
if __name__ == '__main__':
    import timeit

    start = timeit.default_timer()

    logging.basicConfig(level=logging.DEBUG)
    spacy_model_dir = '/data/spacy/sv/ud-suc/version=0.1.1'
    resolver = CoreferenceResolver('sv', spacy_model_dir)
    test_json = open('/data/sv_linked_wikitext_entities.jsonl', 'r')
    doc_count = 100
    for i in range(doc_count):
        line = test_json.readline().strip()
        output = resolver.predict_json(json.loads(line, encoding='utf-8'), print_summary=True)
        print(json.dumps(output, ensure_ascii=False))

    stop = timeit.default_timer()

    print(f"Time for {doc_count} docs: ", stop - start)
    print(f"Avg time per doc: ", (stop - start)/doc_count)
