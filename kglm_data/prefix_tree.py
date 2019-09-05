"""
Implementation of a prefix-tree data structure.

Used to match token sequences in linear time.
"""
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


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
    def __init__(self, fixed=True) -> None:
        self._root = TreeNode()
        self._active = self._root
        self._fixed = fixed

    def __contains__(self, it: Iterable[str]):
        """Sees whether a sequence of elements is a path in the tree."""
        active = self._root
        for elt in it:
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
            it: Iterable[str],
            str_id: str) -> None:
        """Adds a sequence to the tree.

        Args:
            it : ``Iterable[str]``
                A sequence of strings to add as a path in the tree.
            str_id : ``str``
                The identifier for the sequence.
        """
        assert str_id is not None, 'Cannot add a sequence without an `id`'
        active = self._root
        for elt in it:
            if elt not in active:
                active[elt] = TreeNode()
            active = active[elt]
        if active.id is None:
            active.id = str_id
        elif active.id != str_id:
            if self._fixed:
                logger.warning('Collision existing id "%s" with new id "%s"',
                               active.id, str_id)
            else:
                logger.warning('Overwriting existing id "%s" with new id "%s"',
                               active.id, str_id)
                active.id = str_id

