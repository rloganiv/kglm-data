import logging
import unittest

from src.realm_coref import _window, _add_offset, _merge_clusters

logger = logging.getLogger(__name__)


class TestRealmCorefFcts(unittest.TestCase):
    def test_window(self):
        x = [1,2,3,4]
        actual = list(_window(x))
        expected = [[1,2],[2,3],[3,4]]
        self.assertListEqual(actual, expected)

    def test_add_offset(self):
        x = [[1,2],[3,[4]]]
        offset = 2
        actual = _add_offset(x, offset)
        expected = [[3,4],[5,[6]]]
        self.assertListEqual(actual, expected)

    def test_merge_clusters(self):
        clusters_a = [[[1,2],[3,4]],[[5,6]]]
        clusters_b = [[[3,4],[7,8]],[[5,6], [9,10]]]
        all_clusters = [clusters_a, clusters_b]
        actual = _merge_clusters(all_clusters)
        expected = [((1,2),(3,4),(7,8)),((5,6),(9,10))]
        # A bit roundabout, but we need a weird construction to compare sets of
        # sets
        actual_sets = [set(x) for x in actual]
        expected_sets = [set(x) for x in expected]
        failed = False
        for cluster in actual_sets:
            match = False
            for expected_cluster in expected_sets:
                if cluster == expected_cluster:
                    match = True
            self.assertTrue(match)

