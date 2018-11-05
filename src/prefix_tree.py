"""
Implementation of a prefix-tree data structure.

Used to match token sequences in linear time.
"""
import logging
from typing import Tuple, Iterable

logger =  logging.getLogger(__name__)


class TreeNode(dict):
    """A dictionary with an added id field."""
    def __init__(self, id: str=None) -> None:
        super(TreeNode, self).__init__()
        self.id = id

    @property
    def is_terminal(self):
        return self.id is not None


class PrefixTree(object):
    """Basic prefix tree implementation."""
    def __init__(self) -> None:
        self._root = TreeNode()
        self._active = self._root

    def __contains__(self, iter: Iterable[str]):
        """Sees whether a sequence of elements is a path in the tree."""
        active = self._root
        for elt in iter:
            try:
                active = active[elt]
            except KeyError:
                return False
        if active.is_terminal:
            return True
        else:
            return False

    def step(self, x: str) -> str:
        """Looks up whether argument is a child of the active node. If it is
        then the child becomes the active node. If it is not, then an index
        error is raised and the root becomes the active node.

        Args:
            x : ``str``
                Key to look up.

        Returns:
            The child's id.
        """
        try:
            next = self._active[x]
        except KeyError:
            self._active = self._root
            raise IndexError('Could not find "%s" in active branch' % x)
        else:
            self._active = next
        return self._active.id

    def add(self,
            iter: Iterable[str],
            id: str) -> None:
        """Adds a sequence to the tree.

        Args:
            iter : ``Iterable[str]``
                A sequence of strings to add as a path in the tree.
            id : ``str``
                The identifier for the sequence.
        """
        assert id is not None, 'Cannot add a sequence without an `id`'
        active = self._root
        for elt in iter:
            if elt not in active:
                active[elt] = TreeNode()
            active = active[elt]
        if active.id != id and active.id is not None:
            logger.warning('Overwriting existing id "%s" with new id "%s"',
                           active.id, id)
        active.id = id

