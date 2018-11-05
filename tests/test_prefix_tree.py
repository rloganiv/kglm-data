import logging
import unittest

from src.prefix_tree import PrefixTree, TreeNode

logger = logging.getLogger(__name__)


class TestTreeNode(unittest.TestCase):
    def test_is_dict(self):
        node = TreeNode()
        node['a'] = 'b'
        assert node['a'] == 'b'

    def test_id(self):
        node = TreeNode('id')
        assert node.id == 'id'

    def test_is_terminal(self):
        node_a = TreeNode()
        assert not node_a.is_terminal
        node_b = TreeNode('id')
        assert node_b.is_terminal


class TestPrefixTree(unittest.TestCase):
    def test_add(self):
        tree = PrefixTree()
        seq = ['a', 'b', 'c']
        id = 'id'
        tree.add(seq, id)
        assert seq in tree
        for elt in seq:
            out = tree.step(elt)
        assert out == id

    def test_terminal(self):
        tree = PrefixTree()
        seq = ['a', 'b', 'c']
        id = 'id'
        tree.add(seq, id)
        with self.assertRaises(IndexError) as context:
            tree.step('d')

