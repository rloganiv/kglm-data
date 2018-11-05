import logging

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import  Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides

logger = logging.getLogger(__name__)


def _window(l, size=3):
    n = len(l)
    for i in range(n-size+1):
        yield l[i:i+size]


def _add_offset(x, offset):
    if isinstance(x, int):
        return x + offset
    elif isinstance(x, list):
        updated = []
        for element in x:
            updated.append(_add_offset(element, offset))
        return updated
    else:
        raise TypeError('Applied ``_add_offset()`` to something other than a '
                        'list of ints')

def _tuplify_set(x):
    if not isinstance(x, set):
        return x
    else:
        return tuple(_tuplify_set(elt) for elt in x)


def _merge_clusters(all_clusters):
    mapping = dict()
    for clusters in all_clusters:
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
    unique_clusters = set(_tuplify_set(x) for x in mapping.values())
    return list(unique_clusters)


@Predictor.register('realm-coref')
class RealmCorefPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        offset = 0
        all_clusters = []
        for token_window in _window(inputs['tokens']):
            modified_inputs = inputs.copy()
            modified_inputs['tokens'] = token_window
            instance = self._json_to_instance(modified_inputs)
            try:
                prediction = self.predict_instance(instance)
            except:
                logger.error('Bad instance: %s', instance)
                continue
            clusters = _add_offset(prediction['clusters'], offset)
            all_clusters.append(clusters)
            offset += len(token_window[0])
        all_clusters = _merge_clusters(all_clusters)
        inputs['clusters'] = all_clusters
        return inputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict['tokens']
        instance = self._dataset_reader.text_to_instance(tokens)
        return instance

