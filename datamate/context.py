"""
This module handles context management and global settings for Directory objects.
"""

import functools
import inspect
import itertools
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

try:
    from types import NoneType
except ImportError:
    NoneType = type(None)

from typing import Dict, Iterator, Literal, Optional, Union

__all__ = [
    "set_root_dir",
    "get_root_dir",
    "root",
    "set_root_context",
    "delete_if_exists",
    "enforce_config_match",
    "check_size_on_init",
    "get_check_size_on_init",
    "set_verbosity_level",
    "set_scope",
    "get_scope",
    "get_default_scope",
    "reset_scope",
    "context",
]


class context:
    enforce_config_match: ContextVar[bool] = ContextVar(
        "enforce_config_match", default=True
    )
    check_size_on_init: ContextVar[bool] = ContextVar(
        "check_size_on_init", default=False
    )
    verbosity_level: ContextVar[int] = ContextVar("verbosity_level", default=1)
    delete_if_exists: ContextVar[bool] = ContextVar("delete_if_exists", default=False)
    root_dir: ContextVar[Path] = ContextVar("root_dir", default=Path("."))
    within_root_context: ContextVar[bool] = ContextVar(
        "within_root_context", default=False
    )
    scope: ContextVar[Dict[str, type]] = ContextVar("scope", default=None)

    def __setattr__(self, name, value):
        if name in self.__class__.__dict__ and isinstance(
            self.__class__.__dict__[name], ContextVar
        ):
            raise AttributeError(
                f"Cannot overwrite ContextVar {name} directly. Use `.set()` to "
                "update its value."
            )
        super().__setattr__(name, value)


def set_root_dir(root_dir: Optional[Path]) -> None:
    """Set the directory in which to search for Directory objects.

    Args:
        root_dir: Path to set as root directory. If None, uses current directory.
    """
    context.root_dir.set(Path(root_dir) if root_dir is not None else Path("."))


def get_root_dir() -> Path:
    """Return the current Directory search directory.

    Returns:
        Path: Current root directory, defaults to current directory if not set.
    """
    return context.root_dir.get()


def root(
    root_dir: Union[str, Path, NoneType] = None, precedence: Literal[1, 2, 3] = 2
) -> callable:
    """Decorates a callable to fix its individual root directory.

    Args:
        root_dir: Root directory that will be set at execution of the callable.
        precedence: Determines the precedence of this root setting.

            - `1`: Lowest - global and context settings override this
            - `2`: Medium - overrides global but not context settings
            - `3`: Highest - overrides both global and context settings

    Returns:
        callable: Decorated function or class.

    Example:
        ```python
        @root("/path/to/this/individual/dir", precedence=3)
        class MyDirectory(Directory):
            pass

        dir = MyDirectory(...)
        assert dir.path.parent == "/path/to/this/individual/dir"
        ```

    Raises:
        ValueError: If decorator is applied to anything other than function or class.
    """

    if precedence not in [1, 2, 3]:
        raise ValueError("Precedence must be 1, 2, or 3")

    def decorator(callable):
        if inspect.isfunction(callable):

            @functools.wraps(callable)
            def function(*args, **kwargs):
                _root_dir = get_root_dir()
                within_context = context.within_root_context.get(False)

                if root_dir is not None and (
                    precedence == 3
                    or (precedence == 2 and not within_context)
                    or precedence == 1
                    and not within_context
                ):
                    set_root_dir(root_dir)

                _return = callable(*args, **kwargs)
                set_root_dir(_root_dir)
                return _return

            return function
        elif inspect.isclass(callable):
            new = callable.__new__

            @functools.wraps(callable)
            def function(*args, **kwargs):
                _root_dir = get_root_dir()
                within_context = context.within_root_context.get(False)

                if root_dir is not None and (
                    precedence == 3
                    or (precedence == 2 and not within_context)
                    or precedence == 1
                    and not within_context
                ):
                    set_root_dir(root_dir)

                _return = new(*args, **kwargs)
                set_root_dir(_root_dir)
                return _return

            callable.__new__ = function

            return callable
        else:
            raise ValueError("Decorator can only be applied to functions or classes.")

    return decorator


@contextmanager
def set_root_context(root_dir: Union[str, Path, NoneType] = None) -> None:
    """Set root directory within a context and revert after.

    Args:
        root_dir: Temporary root directory to use within the context.

    Example:
        ```python
        with set_root_context(dir):
            Directory(config)
        ```

    Note:
        Takes precedence over all other methods to control the root directory.
    """
    _root_dir = get_root_dir()
    set_root_dir(root_dir)
    context.within_root_context.set(True)
    try:
        yield
    finally:
        set_root_dir(_root_dir)
        context.within_root_context.set(False)


@contextmanager
def delete_if_exists(enable: bool = True) -> None:
    """Delete directory if it exists within a context and revert after.

    Args:
        enable: Whether to enable directory deletion.

    Example:
        ```python
        with delete_if_exists():
            Directory(config)
        ```
    """
    token = context.delete_if_exists.set(enable)
    try:
        yield
    finally:
        context.delete_if_exists.reset(token)


def enforce_config_match(enforce: bool) -> None:
    """Enforce error if configs are not matching.

    Args:
        enforce: Whether to enforce config matching.

    Note:
        Configs are compared when initializing a directory from an existing path
        and configuration.
    """
    context.enforce_config_match.set(enforce)


def check_size_on_init(enforce: bool) -> None:
    """Switch size warning on/off.

    Args:
        enforce: Whether to check directory size on initialization.

    Note:
        Checking the size of a directory is slow, therefore this should be used
        only consciously.
    """
    context.check_size_on_init.set(enforce)


def get_check_size_on_init() -> bool:
    return context.check_size_on_init.get()


def set_verbosity_level(level: Literal[0, 1, 2]) -> None:
    """Set verbosity level of representation for Directory objects.

    Args:
        level: Verbosity level where:

            - `0`: Only top level directory name and last modified date
            - `1`: Maximum 2 levels and 25 lines
            - `2`: All directories and files
    """
    if level not in [0, 1, 2]:
        raise ValueError("Verbosity level must be 0, 1, or 2")
    context.verbosity_level.set(level)


def set_scope(scope: Optional[Dict[str, type]]) -> None:
    """Set the scope used for "type" field resolution.

    Args:
        scope: Dictionary mapping type names to type objects.
    """
    context.scope.set(scope)


def get_scope() -> Dict[str, type]:
    """Return the scope used for "type" field resolution.

    Returns:
        Dict[str, type]: Current scope or default scope if not set.
    """
    return (
        context.scope.get() if context.scope.get() is not None else get_default_scope()
    )


def get_default_scope(cls: Optional[object] = None) -> Dict[str, type]:
    """Return the default scope used for "type" field resolution.

    Args:
        cls: Base class to use for scope generation.

    Returns:
        Dict[str, type]: Mapping of class names to class objects.
    """
    from .directory import Directory

    def subclasses(t: type) -> Iterator[type]:
        yield from itertools.chain([t], *map(subclasses, t.__subclasses__()))

    cls = cls or Directory
    scope: Dict[str, type] = {}
    for t in subclasses(cls):
        scope[t.__qualname__] = t
    return scope


def reset_scope(cls: Optional[object] = None) -> None:
    """Reset the scope to the default scope.

    Args:
        cls: Base class to use for scope generation.
    """
    set_scope(get_default_scope(cls))
