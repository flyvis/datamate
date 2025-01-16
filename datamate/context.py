"""
This module handles context management and global settings for Directory objects.
"""

import threading
from pathlib import Path
from types import NoneType
from typing import Optional, Union, Dict, Literal, Iterator
from contextlib import contextmanager
import itertools
import inspect
import functools

context = threading.local()
context.enforce_config_match = True
context.check_size_on_init = False
context.verbosity_level = 1
context.delete_if_exists = False


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


def set_root_dir(root_dir: Optional[Path]) -> None:
    """Set the directory in which to search for Directory objects.

    Args:
        root_dir: Path to set as root directory. If None, uses current directory.
    """
    context.root_dir = Path(root_dir) if root_dir is not None else Path(".")


def get_root_dir() -> Path:
    """Return the current Directory search directory.

    Returns:
        Path: Current root directory, defaults to current directory if not set.
    """
    return getattr(context, "root_dir", Path("."))


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

    def decorator(callable):
        if inspect.isfunction(callable):

            @functools.wraps(callable)
            def function(*args, **kwargs):
                _root_dir = get_root_dir()
                within_context = getattr(context, "within_root_context", False)

                if root_dir is not None:
                    if precedence == 3 or (precedence == 2 and not within_context):
                        set_root_dir(root_dir)
                    elif precedence == 1 and not within_context:
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
                within_context = getattr(context, "within_root_context", False)

                if root_dir is not None:
                    if precedence == 3 or (precedence == 2 and not within_context):
                        set_root_dir(root_dir)
                    elif precedence == 1 and not within_context:
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
    context.within_root_context = True
    try:
        yield
    finally:
        set_root_dir(_root_dir)
        context.within_root_context = False


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
    context.delete_if_exists = enable
    try:
        yield
    finally:
        context.delete_if_exists = False


def enforce_config_match(enforce: bool) -> None:
    """Enforce error if configs are not matching.

    Args:
        enforce: Whether to enforce config matching.

    Note:
        Configs are compared when initializing a directory from an existing path
        and configuration.
    """
    context.enforce_config_match = enforce


def check_size_on_init(enforce: bool) -> None:
    """Switch size warning on/off.

    Args:
        enforce: Whether to check directory size on initialization.

    Note:
        Checking the size of a directory is slow, therefore this should be used
        only consciously.
    """
    context.check_size_on_init = enforce


def get_check_size_on_init() -> bool:
    return context.check_size_on_init


def set_verbosity_level(level: Literal[0, 1, 2]) -> None:
    """Set verbosity level of representation for Directory objects.

    Args:
        level: Verbosity level where:
            - `0`: Only top level directory name and last modified date
            - `1`: Maximum 2 levels and 25 lines
            - `2`: All directories and files
    """
    context.verbosity_level = level


def set_scope(scope: Optional[Dict[str, type]]) -> None:
    """Set the scope used for "type" field resolution.

    Args:
        scope: Dictionary mapping type names to type objects.
    """
    if hasattr(context, "scope"):
        del context.scope
    if scope is not None:
        context.scope = scope


def get_scope() -> Dict[str, type]:
    """Return the scope used for "type" field resolution.

    Returns:
        Dict[str, type]: Current scope or default scope if not set.
    """
    return context.scope if hasattr(context, "scope") else get_default_scope()


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
