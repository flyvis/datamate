"""
Module for handling nested dictionary-like objects with attribute access support.

This module provides the `Namespace` class and related utilities for working with
nested dictionary-like objects that support both attribute and key-based access.
"""

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    get_origin,
    Iterator,
    Union,
    Tuple,
)
from copy import copy, deepcopy
from numpy import ndarray
from pathlib import Path

import pandas as pd

__all__ = ["Namespace", "namespacify", "is_disjoint", "is_subset", "is_superset"]

# -- Namespaces ----------------------------------------------------------------


class Namespace(Dict[str, Any]):
    """
    A dictionary subclass supporting both attribute and item access.

    Attributes:
        __dict__ (Dict[str, Any]): The underlying dictionary storage.

    Examples:
        ```python
        ns = Namespace({"a": 1, "b": {"c": 2}})
        assert ns.a == 1
        assert ns.b.c == 2
        ```
    """

    def __dir__(self) -> List[str]:
        """Return list of valid attributes including dictionary keys."""
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __getattr__(self, key: str) -> Any:
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, val: object) -> None:
        dict.__setitem__(self, key, val)

    def __delattr__(self, key: str) -> None:
        dict.__delitem__(self, key)

    @property
    def __dict__(self) -> dict:  # type: ignore
        return self

    def __repr__(self) -> str:
        def single_line_repr(elem: object) -> str:
            if isinstance(elem, list):
                return "[" + ", ".join(map(single_line_repr, elem)) + "]"
            elif isinstance(elem, Namespace):
                return (
                    f"{elem.__class__.__name__}("
                    + ", ".join(f"{k}={single_line_repr(v)}" for k, v in elem.items())
                    + ")"
                )
            else:
                return repr(elem).replace("\n", " ")

        def repr_in_context(elem: object, curr_col: int, indent: int) -> str:
            sl_repr = single_line_repr(elem)
            if len(sl_repr) <= 80 - curr_col:
                return sl_repr
            elif isinstance(elem, list):
                return (
                    "[\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        repr_in_context(e, indent + 2, indent + 2) for e in elem
                    )
                    + "\n"
                    + " " * indent
                    + "]"
                )
            elif isinstance(elem, Namespace):
                return (
                    f"{elem.__class__.__name__}(\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        f"{k} = " + repr_in_context(v, indent + 5 + len(k), indent + 2)
                        for k, v in elem.items()
                    )
                    + "\n"
                    + " " * indent
                    + ")"
                )
            else:
                return repr(elem)

        return repr_in_context(self, 0, 0)

    def __eq__(self, other):
        return all_true(compare(namespacify(self), namespacify(other)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def without(self, key: str) -> "Namespace":
        """
        Return a copy of the namespace without the specified key.

        Args:
            key: Key to remove from the namespace.

        Returns:
            New namespace without the specified key.
        """
        _copy = self.deepcopy()
        _copy.pop(key)
        return _copy

    def is_superset(self, other):
        return is_subset(self, other)

    def is_subset(self, other: Union[Dict, "Namespace"]) -> bool:
        """
        Check if this namespace is a subset of another.

        Args:
            other: The potential superset to compare against.

        Returns:
            True if this namespace is a subset of other.
        """
        return is_superset(other, self)

    def is_disjoint(self, other_dict):
        """
        Check whether another dictionary is disjoint with respect to this one.

        Two dictionaries are considered disjoint if they have no common keys.

        Parameters:
        other_dict (dict): The other dictionary to check for disjointness.

        Returns:
        bool: True if the other dictionary is disjoint with respect to this one,
              False otherwise.
        """
        return is_disjoint(self, other_dict)

    def to_df(self, name: str = "", seperator: str = ".") -> "pd.DataFrame":  # type: ignore
        """
        Convert namespace to flattened DataFrame.

        Args:
            name: Column name for the resulting DataFrame.
            seperator: Character to use for separating nested keys.

        Returns:
            Flattened DataFrame representation.
        """
        as_dict = self.to_dict()  # namespace need deepcopy method
        df = pd.json_normalize(as_dict, sep=seperator).T
        if name:
            df = df.rename({0: name}, axis=1)
        return df

    def diff(
        self, other: "Namespace", name1: str = "self", name2: str = "other"
    ) -> "Namespace":
        """
        Compare two namespaces and return their differences.

        Args:
            other: The namespace to compare against.
            name1: Label for the current namespace in the diff output.
            name2: Label for the other namespace in the diff output.

        Returns:
            A namespace containing the differences, with + indicating additions,
            - indicating deletions, and ≠ indicating changes.

        Examples:
            ```python
            ns1 = Namespace({"a": 1, "b": 2})
            ns2 = Namespace({"b": 3, "c": 4})
            diff = ns1.diff(ns2)
            # Returns: {
            #   "self": ["+a: 1", "≠b: 2", "-c"],
            #   "other": ["-a", "≠b: 3", "+c: 4"]
            # }
            ```
        """
        if self is None or other is None:
            return Namespace({name1: self, name2: other})

        _self = namespacify(self)
        other = namespacify(other)
        diff1: List[str] = []
        diff2: List[str] = []
        diff = {name1: diff1, name2: diff2}

        def _diff(self: Namespace, other: Namespace, parent: str = "") -> None:
            for k, v in self.items():
                if k not in other:
                    _diff1 = f"+{parent}.{k}: {v}" if parent else f"+{k}: {v}"
                    _diff2 = f"-{parent}.{k}" if parent else f"-{k}"
                    diff1.append(_diff1)
                    diff2.append(_diff2)
                elif v == other[k]:
                    pass
                elif isinstance(v, Namespace):
                    _parent = f"{parent}.{k}" if parent else f"{k}"
                    _diff(v, other[k], parent=_parent)
                else:
                    _diff1 = f"≠{parent}.{k}: {v}" if parent else f"≠{k}: {v}"
                    _diff2 = (
                        f"≠{parent}.{k}: {other[k]}" if parent else f"≠{k}: {other[k]}"
                    )
                    diff1.append(_diff1)
                    diff2.append(_diff2)

            for k, v in other.items():
                if k not in self:
                    _diff1 = f"-{parent}.{k}" if parent else f"-{k}"
                    _diff2 = f"+{parent}.{k}: {v}" if parent else f"+{k}: {v}"
                    diff1.append(_diff1)
                    diff2.append(_diff2)

        _diff(_self, other)
        return namespacify(diff)

    def walk(self) -> Iterator[Tuple[str, Any]]:
        """
        Recursively walk through the namespace and yield key-value pairs.

        Yields:
            Tuples of (key, value) for each item in the namespace, including nested items.

        Examples:
            ```python
            ns = Namespace({"a": 1, "b": {"c": 2}})
            for key, value in ns.walk():
                print(f"{key}: {value}")
            # Prints:
            # a: 1
            # b: {'c': 2}
            # c: 2
            ```
        """
        yield from dict_walk(self)

    def equal_values(self, other: "Namespace") -> bool:
        """
        Compare values recursively with another namespace.

        Args:
            other: The namespace to compare against.

        Returns:
            True if all values match recursively, False otherwise.
        """
        return compare(self, other)

    def copy(self) -> "Namespace":
        """Create a shallow copy of the namespace."""
        return copy(self)

    def deepcopy(self) -> "Namespace":
        """Create a deep copy of the namespace."""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the namespace to a regular dictionary recursively."""
        return to_dict(self)

    def depth(self) -> int:
        """
        Calculate the maximum depth of nested dictionaries.

        Returns:
            The maximum nesting level, where 0 means no nesting.
        """
        return depth(self)

    def pformat(self):
        return pformat(self)

    def all(self):
        return all_true(self)


def namespacify(obj: object) -> Namespace:
    """
    Recursively convert mappings and ad-hoc Namespaces to `Namespace` objects.

    Args:
        obj: The object to convert into a Namespace.

    Returns:
        A new Namespace object with both item and attribute access.

    Raises:
        TypeError: If the object cannot be converted to a Namespace.

    Examples:
        ```python
        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = {"c": 2}

        obj = MyClass()
        ns = namespacify(obj)
        assert ns.a == 1
        assert ns.b.c == 2
        ```
    """
    if isinstance(obj, (type(None), bool, int, float, str, type, bytes)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [namespacify(v) for v in obj]
    elif isinstance(obj, (ndarray)):
        return [namespacify(v.item()) for v in obj]
    elif isinstance(obj, Mapping):
        return Namespace({k: namespacify(obj[k]) for k in obj})
    elif get_origin(obj) is not None:
        return obj
    else:
        try:
            return namespacify(vars(obj))
        except TypeError as e:
            raise TypeError(f"namespacifying {obj} of type {type(obj)}: {e}.") from e


def is_subset(dict1: Union[Dict, Namespace], dict2: Union[Dict, Namespace]) -> bool:
    """
    Check whether dict2 is a subset of dict1.

    Args:
        dict1: The superset dictionary.
        dict2: The subset dictionary.

    Returns:
        True if dict2 is a subset of dict1, False otherwise.

    Examples:
        ```python
        d1 = {"a": 1, "b": {"c": 2, "d": 3}}
        d2 = {"b": {"c": 2}}
        assert is_subset(d1, d2) == True
        ```
    """
    for key, value in dict2.items():
        if key not in dict1:
            return False
        if isinstance(value, dict):
            if not is_subset(dict1[key], value):
                return False
        else:
            if dict1[key] != value:
                return False
    return True


def is_superset(dict1: Union[Dict, Namespace], dict2: Union[Dict, Namespace]) -> bool:
    """
    Check whether dict2 is a superset of dict1.

    Args:
        dict1: The subset dictionary.
        dict2: The superset dictionary.

    Returns:
        True if dict2 is a superset of dict1, False otherwise.
    """
    return is_subset(dict2, dict1)


def is_disjoint(dict1: Union[Dict, Namespace], dict2: Union[Dict, Namespace]) -> bool:
    """
    Check whether two dictionaries are disjoint.

    Args:
        dict1: First dictionary to compare.
        dict2: Second dictionary to compare.

    Returns:
        True if the dictionaries have no common keys at any nesting level.

    Examples:
        ```python
        d1 = {"a": 1, "b": {"c": 2}}
        d2 = {"d": 3, "e": {"f": 4}}
        assert is_disjoint(d1, d2) == True
        ```
    """
    dict1_keys = set(key for key, _ in dict_walk(dict1))
    dict2_keys = set(key for key, _ in dict_walk(dict2))
    return dict1_keys.isdisjoint(dict2_keys)


def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a Namespace or nested structure to a regular dictionary.

    Args:
        obj: Object to convert to a dictionary.

    Returns:
        A dictionary representation of the input object.
    """
    if isinstance(obj, dict):
        return dict((k, to_dict(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    elif isinstance(obj, Namespace):
        return dict((k, to_dict(v)) for k, v in obj.items())
    else:
        return obj


def depth(obj: Union[Dict, Namespace]) -> int:
    """
    Calculate the maximum depth of nested dictionaries.

    Args:
        obj: Dictionary or Namespace to measure depth of.

    Returns:
        Maximum nesting level, where 0 means no nesting.

    Examples:
        ```python
        d = {"a": 1, "b": {"c": {"d": 2}}}
        assert depth(d) == 3
        ```
    """
    if isinstance(obj, (dict, Namespace)):
        return 1 + (max(map(depth, obj.values())) if obj else 0)
    return 0


def pformat(obj: Any) -> str:
    """
    Pretty format a Namespace or dictionary for display.

    Args:
        obj: Object to format.

    Returns:
        String representation of the object with proper indentation.
    """
    import pprint

    pretty_printer = pprint.PrettyPrinter(depth=100)
    return pretty_printer.pformat(obj)


def compare(obj1: Any, obj2: Any) -> Union[bool, "Namespace"]:
    """
    Type agnostic comparison for basic types and nested dictionaries.

    Args:
        obj1: First object to compare.
        obj2: Second object to compare.

    Returns:
        Boolean for simple types, Namespace of comparison results for complex types.

    Examples:
        ```python
        ns1 = Namespace({"a": 1, "b": {"c": 2}})
        ns2 = Namespace({"a": 1, "b": {"c": 3}})
        result = compare(ns1, ns2)
        assert result.a == True
        assert result.b.c == False
        ```
    """
    if isinstance(obj1, (type(None), bool, int, float, str, type)) and isinstance(
        obj2, (type(None), bool, int, float, str, type)
    ):
        return obj1 == obj2
    elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        return False if len(obj1) != len(obj2) else obj1 == obj2
    elif isinstance(obj1, (ndarray)) and isinstance(obj2, (ndarray)):
        return compare(obj1.tolist(), obj2.tolist())
    elif isinstance(obj1, Mapping) and isinstance(obj2, Mapping):
        _obj1, _obj2 = obj1.deepcopy(), obj2.deepcopy()
        out = {}
        for key in (
            set(_obj1.keys())
            .difference(set(_obj2.keys()))
            .union(set(_obj2.keys()).difference(set(_obj1.keys())))
        ):
            out[key] = False
            _obj1.pop(key, None)
            _obj2.pop(key, None)
        for k in _obj1:
            out[k] = compare(_obj1[k], obj2[k])
        return Namespace(out)
    elif not isinstance(obj1, type(obj2)):
        return False


def all_true(obj: Any) -> bool:
    """
    Check if all elements in a nested structure evaluate to True.

    Args:
        obj: Object to evaluate, can be nested structure.

    Returns:
        True if bool(element) is True for all elements in nested obj.

    Raises:
        TypeError: If object cannot be evaluated.

    Examples:
        ```python
        ns = Namespace({"a": True, "b": {"c": 1, "d": "text"}})
        assert all_true(ns) == True
        ```
    """
    if isinstance(obj, (type(None), bool, int, float, str, type, bytes)):
        return bool(obj)
    elif isinstance(obj, Path):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return all([all_true(v) for v in obj])
    elif isinstance(obj, (ndarray)):
        return all([all_true(v.item()) for v in obj])
    elif isinstance(obj, Mapping):
        return all([all_true(obj[k]) for k in obj])
    else:
        try:
            return all_true(vars(obj))
        except TypeError as e:
            raise TypeError(f"all {obj} of type {type(obj)}: {e}.") from e


def dict_walk(dictionary: Union[Dict, Namespace]) -> Iterator[Tuple[str, Any]]:
    """
    Recursively walk through a nested dictionary and yield key-value pairs.

    Args:
        dictionary: Dictionary or Namespace to traverse.

    Yields:
        Tuple of (key, value) for each item, including nested items.

    Examples:
        ```python
        d = {"a": 1, "b": {"c": 2}}
        for key, value in dict_walk(d):
            print(f"{key}: {value}")
        # Prints:
        # a: 1
        # b: {'c': 2}
        # c: 2
        ```
    """
    for key, value in dictionary.items():
        yield (key, value)
        if isinstance(value, dict):
            yield from dict_walk(value)
