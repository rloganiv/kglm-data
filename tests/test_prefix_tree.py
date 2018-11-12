import logging
import unittest

from src.prefix_tree import PrefixTree, TreeNode

logger = logging.getLogger(__name__)


class TestTreeNode(unittest.TestCase):
    def test_is_dict(self):
        node = TreeNode()
        node['a'] = 'b'
        self.assertEqual(node['a'], 'b')

    def test_id(self):
        node = TreeNode('id')
        self.assertEqual(node.id, 'id')

    def test_is_terminal(self):
        node_a = TreeNode()
        self.assertFalse(node_a.is_terminal)
        node_b = TreeNode('id')
        self.assertTrue(node_b.is_terminal)


class TestPrefixTree(unittest.TestCase):
    def test_add(self):
        tree = PrefixTree()
        seq = ['a', 'b', 'c']
        id = 'id'
        tree.add(seq, id)
        assert seq in tree
        for elt in seq:
            out = tree.step(elt)
        self.assertEqual(out, id)

    def test_terminal(self):
        tree = PrefixTree()
        seq = ['a', 'b', 'c']
        id = 'id'
        tree.add(seq, id)
        with self.assertRaises(IndexError) as context:
            tree.step('d')

