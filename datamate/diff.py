"""
This module provides functionality for comparing Directory objects.
"""

from pathlib import Path
from typing import Dict, List, TYPE_CHECKING
import numpy as np
import pandas as pd


from .io import H5Reader

if TYPE_CHECKING:
    from .directory import Directory


# -- Directory comparison ------------------------------------------------------


class DirectoryDiff:
    """Compare two directories for equality or differences.

    Attributes:
        directory1: First directory to compare.
        directory2: Second directory to compare.
        name1: Name identifier for the first directory.
        name2: Name identifier for the second directory.

    Examples:
        ```python
        dir1 = Directory("path/to/dir1")
        dir2 = Directory("path/to/dir2")
        diff = DirectoryDiff(dir1, dir2)

        # Check if directories are equal
        is_equal = diff.equal()

        # Get differences
        differences = diff.diff()
        ```
    """

    def __init__(
        self,
        directory1: "Directory",
        directory2: "Directory",
        name1: str = None,
        name2: str = None,
    ):
        self.directory1 = directory1
        self.directory2 = directory2
        self.name1 = name1 or self.directory1.path.name
        self.name2 = name2 or self.directory2.path.name

    def equal(self, fail: bool = False) -> bool:
        """Return True if the directories are equal.

        Args:
            fail: If True, raise AssertionError when directories differ.

        Returns:
            bool: True if directories are equal, False otherwise.

        Raises:
            AssertionError: If directories differ and `fail=True`.
        """
        try:
            assert_equal_directories(self.directory1, self.directory2)
            return True
        except AssertionError as e:
            if fail:
                raise AssertionError from e
            return False

    def diff(self, invert: bool = False) -> Dict[str, List[str]]:
        """Return differences between the directories.

        Args:
            invert: If True, swap the order of comparison.

        Returns:
            Dict[str, List[str]]: Dictionary containing differences, keyed by directory names.
        """
        if invert:
            return self._diff_directories(self.directory2, self.directory1)
        return self._diff_directories(self.directory1, self.directory2)

    def config_diff(self) -> Dict[str, List[str]]:
        """Return the differences between the configurations of the directories."""
        return self.directory1.config.diff(
            self.directory2.config, name1=self.name1, name2=self.name2
        )

    def _diff_directories(
        self, dir1: "Directory", dir2: "Directory", parent=""
    ) -> Dict[str, List[str]]:
        from .directory import Directory

        diffs = {self.name1: [], self.name2: []}

        keys1 = set(dir1.keys())
        keys2 = set(dir2.keys())

        # Check for keys only in dir1
        for key in keys1 - keys2:
            val = dir1[key]
            if isinstance(val, H5Reader):
                val = val[()]
            diffs[self.name1].append(self._format_diff("+", key, val, parent))
            diffs[self.name2].append(self._format_diff("-", key, val, parent))

        # Check for keys only in dir2
        for key in keys2 - keys1:
            val = dir2[key]
            if isinstance(val, H5Reader):
                val = val[()]
            diffs[self.name2].append(self._format_diff("+", key, val, parent))
            diffs[self.name1].append(self._format_diff("-", key, val, parent))

        # Check for keys present in both
        for key in keys1 & keys2:
            val1 = dir1[key]
            val2 = dir2[key]
            if isinstance(val1, Directory) and isinstance(val2, Directory):
                child_diffs = self._diff_directories(
                    val1, val2, f"{parent}.{key}" if parent else key
                )
                diffs[self.name1].extend(child_diffs[self.name1])
                diffs[self.name2].extend(child_diffs[self.name2])

            elif isinstance(val1, H5Reader) and isinstance(val2, H5Reader):
                val1 = val1[()]
                val2 = val2[()]
                equal = np.array_equal(val1, val2)
                equal = equal & isinstance(val1, type(val2))
                equal = equal & (val1.dtype == val2.dtype)
                if not equal:
                    diffs[self.name1].append(self._format_diff("≠", key, val1, parent))
                    diffs[self.name2].append(self._format_diff("≠", key, val2, parent))

            elif isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
                equal = val1.equals(val2)
                if not equal:
                    diffs[self.name1].append(self._format_diff("≠", key, val1, parent))
                    diffs[self.name2].append(self._format_diff("≠", key, val2, parent))

            elif val1 != val2:
                diffs[self.name1].append(self._format_diff("≠", key, val1, parent))
                diffs[self.name2].append(self._format_diff("≠", key, val2, parent))

        return diffs

    def _format_diff(self, symbol, key, value, parent):
        full_key = f"{parent}.{key}" if parent else key
        return f"{symbol}{full_key}: {value}"


def assert_equal_attributes(directory: "Directory", target: "Directory") -> None:
    """Assert that two directories have equal attributes.

    Args:
        directory: First directory to compare.
        target: Second directory to compare.

    Raises:
        AssertionError: If directories have different attributes.
    """
    if directory.path == target.path:
        return
    assert isinstance(directory, type(target))
    assert directory._config == target._config
    assert directory.meta == target.meta
    assert directory.__doc__ == target.__doc__
    assert directory.path.exists() == target.path.exists()


def assert_equal_directories(directory: "Directory", target: "Directory") -> None:
    """Assert that two directories are equal.

    Args:
        directory: First directory to compare.
        target: Second directory to compare.

    Raises:
        AssertionError: If directories differ in structure or content.
    """
    from .directory import Directory

    assert_equal_attributes(directory, target)

    assert len(directory) == len(target)
    assert len(list(directory)) == len(list(target))

    keys1 = set(directory.keys())
    keys2 = set(target.keys())
    assert keys1 == keys2

    for k in keys1 & keys2:
        assert k in directory and k in target
        assert k in list(directory) and k in list(target)
        assert hasattr(directory, k) and hasattr(target, k)

        v1 = directory[k]
        v2 = target[k]

        if isinstance(v1, Directory):
            assert isinstance(v2, Directory)
            assert isinstance(getattr(directory, k), Directory) and isinstance(
                getattr(target, k), Directory
            )
            assert_equal_directories(v1, v2)
            assert_equal_directories(getattr(directory, k), v1)
            assert_equal_directories(getattr(target, k), v2)

        elif isinstance(v1, Path):
            assert isinstance(v2, Path)
            assert isinstance(getattr(directory, k), Path) and isinstance(
                getattr(target, k), Path
            )
            assert v1.read_bytes() == v2.read_bytes()
            assert getattr(directory, k).read_bytes() == v1.read_bytes()
            assert getattr(target, k).read_bytes() == v2.read_bytes()

        elif isinstance(v1, pd.DataFrame):
            assert isinstance(v2, pd.DataFrame)
            assert isinstance(getattr(directory, k), pd.DataFrame) and isinstance(
                getattr(target, k), pd.DataFrame
            )
            assert v1.equals(v2)
            assert getattr(directory, k).equals(v1)
            assert getattr(target, k).equals(v2)

        else:
            assert isinstance(v1, H5Reader)
            assert isinstance(v2, H5Reader)
            assert isinstance(getattr(directory, k), H5Reader) and isinstance(
                getattr(target, k), H5Reader
            )
            assert np.array_equal(v1[()], v2[()])
            assert np.array_equal(getattr(directory, k)[()], v1[()])
            assert np.array_equal(getattr(target, k)[()], v2[()])
            assert v1.dtype == v2.dtype
            assert getattr(directory, k).dtype == v1.dtype
            assert getattr(target, k).dtype == v2.dtype
