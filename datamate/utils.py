"""Utility functions for Directory objects."""

import contextlib
import datetime
import itertools
import os
import warnings
from numbers import Number
from pathlib import Path
from typing import Mapping, Union

import numpy as np

__all__ = ["check_size", "byte_to_str", "tree"]


def check_size(
    path: Path, warning_at: int = 20 * 1024**3, print_size: bool = False
) -> int:
    """Check and optionally print directory size, warning if it exceeds threshold.

    Args:
        path: Directory path to check.
        warning_at: Size threshold in bytes to trigger warning.
        print_size: Whether to print the directory size.

    Returns:
        int: Size of directory in bytes.

    Raises:
        FileNotFoundError: If path doesn't exist.
    """

    def sizeof_fmt(num: float, suffix: str = "B") -> str:
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    def get_size(start_path: Path) -> int:
        total_size = 0
        for dirpath, _, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    with contextlib.suppress(FileNotFoundError):
                        total_size += os.path.getsize(fp)
        return total_size

    size_in_bytes = get_size(path)
    if print_size:
        print(f"{sizeof_fmt(size_in_bytes)}")
    if size_in_bytes >= warning_at:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                f"This directory {path.name} occupies {sizeof_fmt(size_in_bytes)} "
                + "of disk space.",
                ResourceWarning,
                stacklevel=2,
            )
    return size_in_bytes


def tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
    last_modified: bool = False,
    not_exists_message: str = "path does not exist",
    permission_denied_message: str = "permission denied",
    verbose: bool = True,
) -> str:
    """Generate a visual tree structure of a directory.

    Args:
        dir_path: Root directory to start tree from.
        level: Maximum depth to traverse (-1 for unlimited).
        limit_to_directories: Only show directories, not files.
        length_limit: Maximum number of lines to output.
        last_modified: Show last modification time of root directory.
        not_exists_message: Message to show when path doesn't exist.
        permission_denied_message: Message to show when permission is denied.
        verbose: Include summary statistics in output.

    Returns:
        str: Formatted tree structure as string.

    Example:
        ```python
        print(tree(Path("./my_directory"), level=2))
        ```

    Based on: https://stackoverflow.com/a/59109706
    """
    # prefix components:
    space = "    "
    branch = "│   "
    # pointers:
    tee = "├── "
    last = "└── "

    tree_string = ""

    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 1

    def inner(dir_path: Path, prefix: str = "", level=-1):
        nonlocal files, directories
        if not level:
            yield prefix + "..."
            return  # 0, stop iterating
        try:
            if limit_to_directories:
                contents = sorted([d for d in dir_path.iterdir() if d.is_dir()])
            else:
                contents = sorted(dir_path.iterdir())
        except PermissionError as e:
            if "[Errno 1]" in str(e):
                contents = [f"({permission_denied_message})"]

        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if isinstance(path, Path):
                if path.is_dir():
                    yield prefix + pointer + path.name + "/"
                    directories += 1
                    extension = branch if pointer == tee else space
                    yield from inner(path, prefix=prefix + extension, level=level - 1)
                elif not limit_to_directories:
                    yield prefix + pointer + path.name
                    files += 1
            else:
                assert path == f"({permission_denied_message})"
                yield prefix + pointer + path

    tree_string += dir_path.name + "/"

    if not dir_path.exists():
        tree_string += f"\n{space}({not_exists_message})"
        return tree_string

    if last_modified:
        timestamp = datetime.datetime.fromtimestamp(dir_path.stat().st_mtime)
        mtime = " - Last modified: {}".format(timestamp.strftime("%B %d, %Y %H:%M:%S"))
        tree_string += mtime
    tree_string += "\n"
    iterator = inner(dir_path, level=level)
    for line in itertools.islice(iterator, length_limit):
        tree_string += line + "\n"

    if verbose:
        if next(iterator, None):
            tree_string += f"... length_limit, {length_limit}, reached,"
        tree_string += (
            f"\ndisplaying: {directories} {'directory' if directories == 1 else 'directories'}"  # noqa: E501
            + (f", {files} files" if files else "")
            + (f", {level} levels." if level >= 1 else "")
        )

    return tree_string


def byte_to_str(
    obj: Union[Mapping, np.ndarray, list, tuple, bytes, str, Number],
) -> Union[Mapping, np.ndarray, list, tuple, str, Number]:
    """Convert byte elements to string types recursively.

    Args:
        obj: Object to convert, can be nested structure containing bytes.

    Returns:
        Converted object with bytes decoded to strings.

    Raises:
        TypeError: If object type cannot be converted.

    Note:
        Function recursively processes nested lists and tuples.
    """
    if isinstance(obj, Mapping):
        return type(obj)({k: byte_to_str(v) for k, v in obj.items()})
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.dtype("S")):
            return obj.astype("U")
        return obj
    elif isinstance(obj, list):
        return [byte_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(byte_to_str(item) for item in obj)
    elif isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, (str, Number)):
        return obj
    else:
        raise TypeError(f"can't cast {obj} of type {type(obj)} to str")
