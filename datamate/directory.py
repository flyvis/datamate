"""
This module exports the `Directory` class, an array- and metadata-friendly view
into a directory.

Instances of the base Directory class have methods to simplify reading/writing
collections of arrays.
"""

import os
import warnings
import itertools
from pathlib import Path
import shutil
from time import sleep
import inspect
from typing import (
    Any,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Dict,
    Union,
    cast,
    get_origin,
    Literal,
    TypeVar,
    Protocol,
    overload,
)
import datetime
from traceback import format_tb
from importlib import import_module


import pandas as pd
from pandas import DataFrame

from .metadata import (
    read_meta,
    write_meta,
    _identify,
)
from .namespaces import (
    Namespace,
    namespacify,
    is_disjoint,
    is_superset,
    to_dict,
)
from .io import (
    ArrayFile,
    H5Reader,
    _read_h5,
    _write_h5,
    _extend_h5,
    _copy_file,
    _copy_dir,
    _extend_file,
    directory_to_dict,
    directory_to_df,
)
from .context import (
    get_root_dir,
    get_scope,
    context,
)
from .utils import check_size, tree
from .diff import DirectoryDiff

__all__ = ["Directory"]

# -- Custom Errors and Warnings ------------------------------------------------


class ConfigWarning(Warning):
    """Warning raised when configuration-related issues occur.

    Typically raised when overriding existing configurations or when configuration
    validation detects potential issues.
    """


class ModifiedWarning(Warning):
    """Warning raised when attempting operations on modified directories.

    Raised when trying to reuse a directory that has been modified after its
    initial construction.
    """


class ModifiedError(Exception):
    """Error raised when modifications to a directory are not allowed.

    Raised when attempting to modify a directory in a way that violates its
    configuration or build status constraints.
    """


class ImplementationWarning(Warning):
    """Warning raised when Directory implementation issues are detected.

    Typically raised when a Directory subclass implementation is missing expected
    components or configurations.
    """


class ImplementationError(Exception):
    """Error raised when Directory implementation is invalid.

    Raised when a Directory subclass implementation violates required patterns
    or constraints.
    """


# -- Directory -----------------------------------------------------------------


class NonExistingDirectory(type):
    """Directory metaclass to allow create non-existing Directory instances."""

    def __call__(cls, *args, **kwargs):
        return cls.__new__(cls, *args, **kwargs)


# Add type variables for better type safety
T = TypeVar("T", bound="Directory")
ConfigType = TypeVar("ConfigType", bound=Dict[str, Any])


class Directory(metaclass=NonExistingDirectory):
    """Array- and metadata-friendly view into a directory.

    Provides a dictionary-like interface for working with arrays and metadata stored in
    a directory structure.

    Args:
        path: Path at which the Directory is/should be stored. Can be relative to current
            `root_dir`.
        config: Configuration dictionary. When including a `type` field, indicates the
            Directory type to search for and construct.

    Attributes:
        path: Path where Directory is stored.
            Type: `pathlib.Path`
        config: Directory configuration.
            Type: `Config`
        meta: Metadata stored in `_meta.yaml`.
            Type: `Namespace`
        status: Build status from metadata.
            Type: `Literal["running", "done", "stopped"]`
        parent: Parent directory.
            Type: `Directory`

    Valid constructors:
        ```python
        # Auto-name relative to root_dir:
        Directory()
        Directory(config={"type": "MyType"})

        # Name relative to root_dir or absolute:
        Directory("/path/to/dir")
        Directory("/path/to/dir", config={"type": "MyType"})
        ```

    After instantiation, Directories act as string-keyed mutable dictionaries containing:
    - ArrayFiles: Single-entry HDF5 files in SWMR mode
    - Paths: Non-array files in other formats
    - Directories: Subdirectories

    Array-like numeric and byte-string data written via `__setitem__`, `__setattr__`, or
    `extend` is stored as an array file.

    Example:
        ```python
        # Create directory with arrays
        dir = Directory("my_data")
        dir["array1"] = np.array([1,2,3])
        dir.array2 = np.array([[4,5],[6,7]])

        # Access data
        arr1 = dir["array1"]  # Returns array([1,2,3])
        arr2 = dir.array2     # Returns array([[4,5],[6,7]])
        ```
    """

    class Config(Protocol):
        """Protocol defining the configuration interface for Directory classes.

        This protocol defines the structure of the `config` argument to the
        `Directory` constructor, and provides type hints for the `config`
        attribute of `Directory` instances.

        Note:
            Subclasses should implement this protocol to define their configuration
            interface, or implement `__init__` with typed parameters.
        """

        pass

    path: Path
    config: Config

    @overload
    def __new__(cls: type[T]) -> T: ...

    @overload
    def __new__(cls: type[T], path: Union[str, Path]) -> T: ...

    @overload
    def __new__(cls: type[T], config: ConfigType) -> T: ...

    @overload
    def __new__(cls: type[T], path: Union[str, Path], config: ConfigType) -> T: ...

    def __new__(_type: type[T], *args: object, **kwargs: object) -> T:
        """Implementation of overloaded constructors."""
        path, config = _parse_directory_args(args, kwargs)

        if path is not None and isinstance(path, Path) and path.exists():
            # case 1: path exists and global context is deleting if exists
            if context.delete_if_exists:
                shutil.rmtree(path)
            # case 2: path exists and local kwargs are deleting if exists
            if (
                config is not None
                and "delete_if_exists" in config
                and config["delete_if_exists"]
            ):
                shutil.rmtree(path)

            if config is not None and "delete_if_exists" in config:
                # always remove the deletion flag from the config
                config.pop("delete_if_exists")

        cls = _directory(_type)
        _check_implementation(cls)

        defaults = get_defaults(cls)

        if config is None and defaults:  # and _implements_init(cls):
            # to initialize from defaults if no config or path is provided
            if path is None:
                config = defaults
            # to initialize from defaults if no config and empty path is provided
            elif path is not None and not path.exists():
                config = defaults
            # if a non-empty path is provided, we cannot initialize from defaults
            else:
                pass
        # breakpoint()
        if path is not None and config is None:
            cls = _directory_from_path(cls, _resolve_path(path))
        elif path is None and config is not None:
            cls = _directory_from_config(cls, config)
        elif path is not None and config is not None:
            cls = _directory_from_path_and_config(cls, _resolve_path(path), config)
        elif path is None and config is None:
            if _implements_init(cls):
                # raise ValueError("no configuration provided")
                pass

        if context.check_size_on_init:
            cls.check_size()

        return cls

    def __init__(self) -> None:
        """Implement to compile `Directory` from a configuration.

        Note:
            Subclasses can either implement `Config` to determine the interface,
            types and defaults of `config`, or implement `__init__` with keyword
            arguments. If both are implemented, the config is created from the joined
            interface as long as defaults are not conflicting.
        """
        pass

    def __init_subclass__(cls, **kwargs) -> None:
        """Initializes a Directory subclass.

        Automatically generates documentation for the subclass.

        Args:
            **kwargs: Additional keyword arguments passed to parent __init_subclass__
        """
        super().__init_subclass__(**kwargs)
        cls.__doc__ = _auto_doc(cls)

    @property
    def meta(self) -> Namespace:
        """The metadata stored in `{self.path}/_meta.yaml`."""
        return read_meta(self.path)

    @property
    def config(self) -> Config:
        """The directory configuration."""
        return self.meta.config or self._config

    @config.setter
    def config(self, value: Config) -> None:
        self.__manual_config(value)

    @property
    def status(self) -> Literal["running", "done", "stopped"]:
        """The build status from metadata."""
        return self.meta.status

    @property
    def size(self) -> int:
        """Total size of directory in bytes."""
        return check_size(self.path, warning_at=float("inf"), print_size=False)

    @property
    def is_empty(self) -> bool:
        """Whether directory contains any files."""
        return len(self) == 0

    @property
    def modified(self) -> bool:
        """Whether directory has been modified after initialization."""
        return getattr(self.meta, "modified", False)

    # -- MutableMapping methods ----------------------------

    def __len__(self) -> int:
        """Returns the number of public files in `self.path`.

        Non-public files (files whose names start with "_") are not counted.

        Returns:
            Number of public files in the directory
        """
        return sum(1 for _ in self.path.glob("[!_]*"))

    def __iter__(self) -> Iterator[str]:
        """Yields field names corresponding to the public files in `self.path`.

        Entries it understands (subdirectories and HDF5 files) are yielded
        without extensions. Non-public files (files whose names start with "_")
        are ignored.

        Yields:
            Field names for each public file
        """
        for p in self.path.glob("[!_]*"):
            yield p.name.rpartition(".")[0] if p.suffix in [".h5", ".csv"] else p.name

    def __copy__(self) -> "Directory":
        """Creates a shallow copy of the directory.

        Returns:
            New `Directory` instance pointing to the same path
        """
        return Directory(self.path)

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def keys(self) -> Iterator[str]:
        """Returns an iterator over public file names in the directory.

        Returns:
            Iterator yielding public file names
        """
        return self.__iter__()

    def items(self) -> Iterator[Tuple[str, ArrayFile]]:
        """Returns an iterator over (key, value) pairs in the directory.

        Returns:
            Iterator yielding tuples of (filename, file content)
        """
        for key in self.keys():
            yield (key, self[key])

    @classmethod
    def from_df(
        cls: type[T], df: DataFrame, dtypes: Dict[str, Any], *args, **kwargs
    ) -> T:
        """Create a Directory from a DataFrame by splitting into column arrays.

        Each column is stored as a separate HDF5 array with specified dtype.
        This is different from storing the DataFrame directly, which uses CSV format.

        Args:
            df: Source DataFrame
            dtypes: Dictionary mapping column names to numpy dtypes
            *args: Additional arguments passed to `Directory` constructor
            **kwargs: Additional keyword arguments passed to `Directory` constructor

        Returns:
            `Directory` with each column stored as a separate array

        Examples:
            ```python
            df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
            dtypes = {'a': np.int64, 'b': 'S'}

            # Store columns as separate arrays
            dir1 = Directory.from_df(df, dtypes)
            # Results in:
            # dir1/
            #   ├── a.h5  # array([1, 2, 3])
            #   └── b.h5  # array([b'x', b'y', b'z'])

            # Store as single CSV
            dir2 = Directory()
            dir2['data'] = df
            # Results in:
            # dir2/
            #   └── data.csv
            ```
        """
        directory = Directory.__new__(Directory, *args, **kwargs)
        directory.update(
            {column: df[column].values.astype(dtypes[column]) for column in df.columns}
        )
        return directory

    def update(self, other: Union[Dict, "Directory"], suffix: str = "") -> None:
        """Updates self with items of other and appends an optional suffix.

        Args:
            other: Dictionary or Directory to copy items from
            suffix: Optional string to append to copied keys
        """
        for key in other:
            if key + suffix not in self:
                self[key + suffix] = other[key]

    def move(self, dst: Union[str, Path]) -> "Directory":
        """Moves directory to new location.

        Args:
            dst: Destination path

        Returns:
            New `Directory` instance at the destination path
        """
        shutil.move(self.path, dst)
        return Directory(dst)

    def rmtree(self, y_n: Optional[str] = None) -> None:
        """Recursively deletes the directory after confirmation.

        Args:
            y_n: Optional pre-supplied confirmation ('y' or 'n'). If not provided,
                will prompt user interactively
        """
        reply = y_n or input(f"delete {self.path} recursively, y/n?")
        if reply.lower() == "y":
            shutil.rmtree(self.path, ignore_errors=True)

    def _rebuild(self, y_n: Optional[str] = None) -> None:
        """Rebuilds the directory by deleting and recreating it.

        Args:
            y_n: Optional pre-supplied confirmation ('y' or 'n'). If not provided,
                will prompt user interactively
        """
        self.rmtree(y_n)
        _build(self)

    def __truediv__(self, other: str) -> Any:
        """Implements path-like division operator for accessing entries.

        Args:
            other: Key to access

        Returns:
            Same as `self[other]`
        """
        return self.__getitem__(other)

    def __getitem__(self, key: str) -> Any:
        """Returns an `ArrayFile`, `Path`, or `Directory` corresponding to `self.path/key`.

        HDF5 files are returned as `ArrayFile`s, other files are returned as `Path`s, and
        directories and nonexistent entries are returned as (possibly empty) `Directory`s.

        Args:
            key: Name of the entry to retrieve

        Returns:
            The requested entry as an appropriate type

        Note:
            Attribute access syntax is also supported, and occurrences of `__` in `key` are
            transformed into `.`, to support accessing encoded files as attributes
            (i.e. `Directory['name.ext']` is equivalent to `Directory.name__ext`).
        """
        # if context.in_memory:
        #     return object.__getattribute__(self, key)

        try:
            # to catch cases where key is an index to a reference to an h5 file.
            # this will yield a TypeError because Path / slice does not work.
            path = self.path / key
        except TypeError as e:
            if not self.path.exists():
                # we wanted to index an H5Dataset but we tried to index a Directory
                # because the H5Dataset does not exist
                raise (
                    FileNotFoundError(
                        f"Indexing {self.path.name} at {key} not possible for"
                        f" Directory at {self.path.parent}. File "
                        f"{self.path.name}.h5 does not exist."
                    )
                )
            raise e

        # Return an array.
        if path.with_suffix(".h5").is_file():
            return _read_h5(path.with_suffix(".h5"))

        # Return a csv
        if path.with_suffix(".csv").is_file():
            return pd.read_csv(path.with_suffix(".csv"))

        # Return the path to a file.
        elif path.is_file():
            return path

        # Return a subrecord
        else:
            return Directory(path)

    def __setitem__(self, key: str, val: object) -> None:
        """
        Writes an `ArrayFile`, `Path`, or `Directory` to `self.path/key`

        `np.ndarray`-like objects are written as `ArrayFiles`, `Path`-like
        objects are written as `Path`s, and string-keyed mappings are
        written as subDirectorys.

        Attribute access syntax is also supported, and occurrences of "__" in
        `key` are transformed into ".", to support accessing encoded files as
        attributes (i.e. `Directory['name.ext'] = val` is equivalent to
        `Directory.name__ext = val`).
        """
        # if context.in_memory:
        #     object.__setattr__(self, key, val)
        #     return

        path = self.path / key

        # Copy an existing file or directory.
        if isinstance(val, Path):
            if os.path.isfile(val):
                _copy_file(path, val)
            elif os.path.isdir(val):
                _copy_dir(path, val)

        # Write a Directory instance
        elif isinstance(val, Directory):
            assert path.suffix == ""
            # Create new directory with same type and config as source
            new_dir = type(val)(path, config=val.config)
            MutableMapping.update(new_dir, val)

        # Write a mapping as a new Directory
        elif isinstance(val, Mapping):
            assert path.suffix == ""
            MutableMapping.update(Directory(path), val)  # type: ignore

        # Write a dataframe.
        elif isinstance(val, pd.DataFrame):  # Use pd.DataFrame explicitly
            assert path.suffix == ""
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            val.to_csv(path.with_suffix(".csv"), index=False)

        # Write an array.
        else:
            assert path.suffix == ""
            if isinstance(val, H5Reader):
                val = val[()]
            try:
                _write_h5(path.with_suffix(".h5"), val)
            except TypeError as err:
                raise TypeError(
                    format_tb(err.__traceback__)[0]
                    + err.args[0]
                    + f"\nYou're trying to store {val} which cannot be converted to h5-file in {path}."
                    + "\nFor reference of supported types, see https://docs.h5py.org/en/stable/faq.html?highlight=types#numpy-object-types"
                    + "\nE.g. NumPy unicode strings must be converted to 'S' strings and back:"
                    + "\nfoo.bar = array.astype('S') to store and foo.bar[:].astype('U') to retrieve."
                ) from None

        if self.config is not None and self.status == "done":
            # Track if a Directory has been modified past __init__
            self._modified_past_init(True)

    def __delitem__(self, key: str) -> None:
        """
        Deletes the entry at `self.path/key`

        Attribute access syntax is also supported, and occurrences of "__" in
        `key` are transformed into ".", to support accessing encoded files as
        attributes (i.e. `del Directory['name.ext']` is equivalent to
        `del Directory.name__ext`).
        """
        # if context.in_memory:
        #     object.__delitem__(self, key)
        #     return
        path = self.path / key

        # Delete an array file.
        if path.with_suffix(".h5").is_file():
            path.with_suffix(".h5").unlink()

        # Delete a csv file.
        if path.with_suffix(".csv").is_file():
            path.with_suffix(".csv").unlink()

        # Delete a non-array file.
        elif path.is_file():
            path.unlink()

        # Delete a Directory.
        else:
            shutil.rmtree(path, ignore_errors=True)

    def __eq__(self, other: object) -> bool:
        """Returns True if `self` and `other` are equal.

        Two Directories are equal if they have the same keys and the same
        values for each key.

        Args:
            other: Object to compare against

        Returns:
            Whether the directories are equal

        Raises:
            ValueError: If comparing `Directory` with incompatible type
        """
        if not isinstance(other, Directory):
            raise ValueError(f"Cannot compare Directory to {type(other)}")

        if self.path == other.path:
            return True

        if self.path != other.path:
            diff = DirectoryDiff(self, other)
            return diff.equal(fail=False)

    def __neq__(self, other: object) -> bool:
        """Returns True if directories are not equal.

        Args:
            other: Object to compare against

        Returns:
            Whether the directories are not equal
        """
        return not self.__eq__(other)

    def diff(self, other: "Directory") -> Dict[str, List[str]]:
        """Returns a dictionary of differences between this directory and another.

        Args:
            other: Directory to compare against

        Returns:
            Dictionary with two keys - the name of self and other. Values are lists of
            strings describing differences between corresponding entries.
        """
        diff = DirectoryDiff(self, other)
        return diff.diff()

    def extend(self, key: str, val: object) -> None:
        """Extends an array, file or directory at the given key.

        Extending arrays performs concatenation along the first axis,
        extending files performs byte-level concatenation, and
        extending directories extends their fields.

        Args:
            key: Name of the entry to extend
            val: Value to append to the existing entry. Can be `np.ndarray`, `Path`,
                `Directory`, or `Mapping`

        Note:
            Files corresponding to `self[key]` are created if they do not already exist.
        """
        # if context.in_memory:
        #     self.__setitem__(key, np.append(self.__getitem__(key), val, axis=0))

        path = self.path / key

        # Append an existing file.
        if isinstance(val, Path):
            assert path.suffix != ""
            _extend_file(path, val)

        # Append a subDirectory.
        elif isinstance(val, (Mapping, Directory)):
            assert path.suffix == ""
            for k in val:
                Directory(path).extend(k, val[k])

        elif isinstance(val, pd.DataFrame):
            assert path.suffix == ""
            if path.with_suffix(".csv").is_file():
                old_df = pd.read_csv(path.with_suffix(".csv"))
                new_df = pd.concat([old_df, val], axis=0)
            else:
                new_df = val
            new_df.to_csv(path.with_suffix(".csv"), index=False)

        # Append an array.
        else:
            assert path.suffix == ""
            if isinstance(val, H5Reader):
                val = val[()]
            _extend_h5(path.with_suffix(".h5"), val)

        if self.config is not None and self.status == "done":
            # Track if a Directory has been modified past __init__
            self._modified_past_init(True)

    # --- Views ---

    def __repr__(self):
        if context.verbosity_level == 1:
            string = tree(
                self.path,
                last_modified=True,
                level=2,
                length_limit=25,
                verbose=True,
                not_exists_message="empty",
            )
        elif context.verbosity_level == 0:
            string = tree(
                self.path,
                last_modified=True,
                level=1,
                length_limit=0,
                verbose=False,
                not_exists_message="empty",
            )
        else:
            string = tree(
                self.path,
                level=-1,
                length_limit=None,
                last_modified=True,
                verbose=True,
                limit_to_directories=False,
            )
        return string

    def tree(
        self,
        level: int = -1,
        length_limit: Optional[int] = None,
        verbose: bool = True,
        last_modified: bool = True,
        limit_to_directories: bool = False,
    ) -> None:
        """Prints a tree representation of the directory structure.

        Args:
            level: Maximum depth to display (-1 for unlimited)
            length_limit: Maximum number of entries to show per directory
            verbose: Whether to show detailed information
            last_modified: Whether to show last modification times
            limit_to_directories: Whether to only show directories
        """
        print(
            tree(
                self.path,
                level=level,
                length_limit=length_limit,
                last_modified=last_modified,
                verbose=verbose,
                limit_to_directories=limit_to_directories,
            )
        )

    # -- Attribute-style element access --------------------

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__") and key.endswith("__"):  # exclude dunder attributes
            return None
        return self.__getitem__(key.replace("__", "."))

    def __setattr__(self, key: str, value: object) -> None:
        # Fix autoreload related effect.
        if key.startswith("__") and key.endswith("__"):
            object.__setattr__(self, key, value)
            return
        # allow manual config writing
        if key == "config":
            self.__manual_config(value)
            return
        self.__setitem__(key.replace("__", "."), value)

    def __delattr__(self, key: str) -> None:
        self.__delitem__(key.replace("__", "."))

    # -- Attribute preemption, for REPL autocompletion -----

    def __getattribute__(self, key: str) -> Any:
        if key in object.__getattribute__(self, "_cached_keys"):
            try:
                object.__setattr__(self, key, self[key])
            except KeyError:
                object.__delattr__(self, key)
                object.__getattribute__(self, "_cached_keys").remove(key)
        return object.__getattribute__(self, key)

    def __dir__(self) -> List[str]:
        for key in self._cached_keys:
            object.__delattr__(self, key)
        self._cached_keys.clear()

        for key in set(self).difference(object.__dir__(self)):
            object.__setattr__(self, key, self[key])
            self._cached_keys.add(key)

        return cast(list, object.__dir__(self))

    # -- Convenience methods

    def __manual_config(self, config, status=None):
        """Overriding config stored in _meta.yaml.

        config (Dict): update for meta.config
        status (str): status if config did not exist before, i.e. _overrid_config
            is used to store a _meta.yaml for the first time instead of build.
        """
        meta_path = self.path / "_meta.yaml"

        current_config = self.config
        config = namespacify(config)
        if current_config is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (
                        f"Overriding config. Diff is:"
                        f'{config.diff(current_config, name1="passed", name2="stored")}'
                    ),
                    ConfigWarning,
                    stacklevel=2,
                )
            write_meta(path=meta_path, config=config, status="manually written")
        else:
            write_meta(path=meta_path, config=config, status=status or self.status)

    def _override_status(self, status: Literal["running", "done", "stopped"]) -> None:
        """Overrides the build status in metadata.

        Args:
            status: New status to set. Must be one of "running", "done", or "stopped"

        Warns:
            `ConfigWarning`: When overriding an existing status
        """
        meta_path = self.path / "_meta.yaml"

        current_status = self.status
        if current_status is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (f"Overriding status {current_status} to {status}"),
                    ConfigWarning,
                    stacklevel=2,
                )
        write_meta(path=meta_path, config=self.config, status=status)

    def _modified_past_init(self, is_modified: bool) -> None:
        """Tracks if a `Directory` has been modified after initialization.

        Updates the metadata file to record modification status.

        Args:
            is_modified: Whether the directory has been modified

        Note:
            This is used to warn users when attempting to reuse a modified directory.
        """
        meta_path = self.path / "_meta.yaml"

        if is_modified:
            write_meta(
                path=meta_path, config=self.config, status=self.status, modified=True
            )

    def check_size(
        self,
        warning_at: int = 20 * 1024**3,  # 20GB
        print_size: bool = False,
        *,
        raise_on_warning: bool = False,
    ) -> int:
        """Checks the total size of the directory.

        Args:
            warning_at: Size in bytes at which to issue a warning
            print_size: Whether to print the directory size
            raise_on_warning: Whether to raise exception instead of warning

        Returns:
            Total size in bytes

        Raises:
            ValueError: If directory size exceeds warning_at and raise_on_warning is True
        """
        size = check_size(self.path, warning_at, print_size)
        if raise_on_warning and size > warning_at:
            raise ValueError(f"Directory size {size} exceeds limit {warning_at}")
        return size

    def to_df(self, dtypes: Optional[Dict[str, Any]] = None) -> DataFrame:
        """Reconstruct a DataFrame from HDF5 column arrays in this directory.

        Combines all equal-length, single-dimensional HDF5 datasets into DataFrame columns.
        Results are cached to avoid expensive recomputation.

        Args:
            dtypes: Optional dictionary mapping column names to numpy dtypes

        Returns:
            `DataFrame` reconstructed from HDF5 column arrays

        Note:
            This is the complement to `from_df()`. While direct DataFrame assignment
            stores as CSV, `from_df()` splits columns into HDF5 arrays which can be
            recombined using this method.
        """
        try:
            return object.__getattribute__(self, "_as_df")
        except AttributeError:
            object.__setattr__(self, "_as_df", directory_to_df(self, dtypes))
            return self.to_df()

    def to_dict(self) -> DataFrame:
        """
        Returns a DataFrame from all equal length, single-dim .h5 datasets in self.path.
        """
        # to cache the dict that is expensive to create.
        try:
            return object.__getattribute__(self, "_as_dict")
        except AttributeError:
            object.__setattr__(self, "_as_dict", directory_to_dict(self))
            return self.to_dict()

    def mtime(self) -> datetime.datetime:
        """Returns the last modification time of the directory.

        Returns:
            Datetime object representing last modification time
        """
        return datetime.datetime.fromtimestamp(self.path.stat().st_mtime)

    @property
    def parent(self) -> "Directory":
        """The parent directory."""
        return Directory(self.path.absolute().parent)

    def _count(self) -> int:
        """Counts number of existing numbered subdirectories.

        Returns:
            Number of subdirectories matching pattern '[0-9a-f]{4}'
        """
        root = self.path
        count = 0
        for i in itertools.count():
            dst = root / f"{i:04x}"
            if dst.exists():
                count += 1
            else:
                return count
        return count

    def _next(self) -> "Directory":
        """Creates next available numbered subdirectory.

        Returns:
            New `Directory` instance for the next available numbered subdirectory

        Raises:
            AssertionError: If the next numbered directory already exists
        """
        root = self.path
        dst = root / f"{self._count():04x}"
        assert not dst.exists()
        return Directory(dst, self.config)

    def _clear_filetype(self, suffix: str) -> None:
        """Deletes all files with given suffix in the current directory.

        Args:
            suffix: File extension to match (e.g. '.h5', '.csv')
        """
        for file in self.path.iterdir():
            if file.is_file() and file.suffix == suffix:
                file.unlink()


# -- Directory construction -----------------------------------------------------


def merge(dict1: Dict, dict2: Dict) -> Dict:
    """Merges two dictionaries with conflict checking.

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Merged dictionary

    Raises:
        ValueError: If dictionaries have conflicting values for the same keys
    """
    merged = {}
    if is_disjoint(dict1, dict2):
        merged.update(dict1)
        merged.update(dict2)
    elif is_superset(dict1, dict2):
        merged.update(dict1)
    elif is_superset(dict2, dict1):
        merged.update(dict2)
    else:
        raise ValueError(f"merge conflict: {dict1} and {dict2}")
    return merged


def get_defaults(cls: Directory) -> Dict[str, Any]:
    """Gets default configuration values for a Directory class.

    Merges defaults from both `Config` class and `__init__` method.

    Args:
        cls: Directory class to get defaults for

    Returns:
        Dictionary of default configuration values

    Raises:
        ValueError: If defaults from Config and __init__ conflict
    """
    try:
        return merge(get_defaults_from_Config(cls), get_defaults_from_init(cls))
    except ValueError as e:
        raise ValueError("conflicting defaults") from e


def get_defaults_from_Config(cls: Union[type, Directory]) -> Dict[str, Any]:
    """Gets default values from a Directory class's Config class.

    Args:
        cls: Directory class or instance to get defaults from

    Returns:
        Dictionary of default values defined in Config class
    """
    cls = cls if isinstance(cls, type) else type(cls)
    if "Config" in cls.__dict__:
        return {
            k: v
            for k, v in cls.Config.__dict__.items()
            if not (k.startswith("_") or (k.startswith("__") and k.endswith("__")))
        }
    return {}


def get_defaults_from_init(cls: Directory) -> Dict[str, Any]:
    """Gets default values from a Directory class's __init__ parameters.

    Args:
        cls: Directory class to get defaults from

    Returns:
        Dictionary of default values from __init__ signature
    """
    cls = cls if isinstance(cls, type) else type(cls)
    signature = inspect.signature(cls.__init__)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default != inspect._empty
    }
    return defaults


def get_annotations(cls: Union[type, Directory]) -> Dict[str, Any]:
    """Gets type annotations for a Directory class.

    Merges annotations from both `Config` class and `__init__` method.

    Args:
        cls: Directory class to get annotations for

    Returns:
        Dictionary of type annotations
    """
    return merge(get_annotations_from_Config(cls), get_annotations_from_init(cls))


def get_annotations_from_Config(cls: Union[type, Directory]) -> Dict[str, Any]:
    """Gets type annotations from a Directory class's Config class.

    Args:
        cls: Directory class or instance to get annotations from

    Returns:
        Dictionary of type annotations defined in Config class
    """
    cls = cls if isinstance(cls, type) else type(cls)
    if "Config" in cls.__dict__:
        annotations = getattr(cls.Config, "__annotations__", {})
        return annotations
    return {}


def get_annotations_from_init(cls: Directory) -> Dict[str, Any]:
    """Gets type annotations from a Directory class's __init__ parameters.

    Args:
        cls: Directory class to get annotations from

    Returns:
        Dictionary of type annotations from __init__ signature
    """
    cls = cls if isinstance(cls, type) else type(cls)
    return {k: v for k, v in cls.__init__.__annotations__.items() if v is not None}


def _parse_directory_args(
    args: Tuple[object, ...], kwargs: Mapping[str, object]
) -> Tuple[Optional[Path], Optional[Mapping[str, object]]]:
    """Parses constructor arguments for Directory class.

    Valid signatures:
    ```python
    Directory()
    Directory(config: Dict[str, object])
    Directory(path: Path)
    Directory(path: Path, config: Dict[str, object])
    Directory(name: str)
    Directory(name: str, config: Dict[str, object])
    ```

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tuple of (path, config)

    Raises:
        TypeError: If arguments don't match any valid signature
    """
    # ()
    if len(args) == 0 and len(kwargs) == 0:
        return None, None

    # (conf)
    elif len(args) == 1 and isinstance(args[0], Mapping) and len(kwargs) == 0:
        return None, dict(args[0])

    # (config=conf)
    elif len(args) == 0 and len(kwargs) == 1 and "config" in kwargs:
        return None, kwargs["config"]

    # (**conf)
    elif len(args) == 0 and len(kwargs) > 0:
        return None, kwargs

    # (path)
    elif len(args) == 1 and isinstance(args[0], Path) and len(kwargs) == 0:
        return Path(args[0]), None

    # (str)
    elif len(args) == 1 and isinstance(args[0], str) and len(kwargs) == 0:
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), None
        root_dir = get_root_dir()
        return root_dir / args[0], None

    # (path, conf)
    elif (
        len(args) == 2
        and isinstance(args[0], Path)
        and isinstance(args[1], Mapping)
        and len(kwargs) == 0
    ):
        return Path(args[0]), dict(args[1])

    # (str, conf)
    elif (
        len(args) == 2
        and isinstance(args[0], str)
        and isinstance(args[1], Mapping)
        and len(kwargs) == 0
    ):
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), dict(args[1])
        root_dir = get_root_dir()
        return root_dir / args[0], dict(args[1])

    # (path, config=conf)
    elif (
        len(args) == 1
        and isinstance(args[0], Path)
        and len(kwargs) == 1
        and "config" in kwargs
    ):
        return Path(args[0]), kwargs["config"]

    # (path, **conf)
    elif len(args) == 1 and isinstance(args[0], Path) and len(kwargs) > 0:
        return Path(args[0]), kwargs

    # (str, config=conf)
    elif (
        len(args) == 1
        and isinstance(args[0], str)
        and len(kwargs) == 1
        and "config" in kwargs
    ):
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), kwargs["config"]
        root_dir = get_root_dir()
        return root_dir / args[0], kwargs["config"]

    # (str, **conf)
    elif len(args) == 1 and isinstance(args[0], str) and len(kwargs) > 0:
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), kwargs
        root_dir = get_root_dir()
        return root_dir / args[0], kwargs

    # <invalid signature>
    else:
        raise TypeError(
            "Invalid argument types for the `Directory` constructor.\n"
            "Valid signatures:\n"
            "\n"
            "    - Directory()\n"
            "    - Directory(config: Dict[str, object])\n"
            "    - Directory(path: Path)\n"
            "    - Directory(path: Path, config: Dict[str, object])\n"
            "    - Directory(name: str)\n"
            "    - Directory(name: str, config: Dict[str, object])"
            "Note, config can also be passed as keyword arguments after "
            "`path` or `name`. `name` is relative to the root directory."
        )


def _implements_init(cls: Directory) -> bool:
    """Checks if a Directory class implements __init__.

    Args:
        cls: Directory class to check

    Returns:
        True if class has a non-pass __init__ implementation
    """
    return inspect.getsource(cls.__init__).split("\n")[-2].replace(" ", "") != "pass"


def _check_implementation(cls: Directory) -> None:
    """Checks if Directory subclass is properly implemented.

    Verifies that classes implementing `__init__` have appropriate configuration.

    Args:
        cls: Directory class to check

    Warns:
        ImplementationWarning: If `__init__` is implemented without configuration
    """
    defaults = get_defaults(cls)
    # check if Config only has annotations, no defaults
    annotations = get_annotations(cls)

    if _implements_init(cls) and not defaults and not annotations:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                (
                    f"The Directory {type(cls)} implements __init__ to write data"
                    " but specifies no configuration."
                ),
                ImplementationWarning,
                stacklevel=2,
            )


def _directory(cls: type) -> Directory:
    """Returns a new Directory at the root of the file tree.

    Args:
        cls: Directory class to instantiate

    Returns:
        New Directory instance with path at root directory
    """
    directory = _forward_subclass(cls, {})
    path = _new_directory_path(type(directory))
    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)
    return directory


def _directory_from_path(cls: Directory, path: Path) -> Directory:
    """
    Return a Directory corresponding to the file tree at `path`.

    An error is raised if the type recorded in `_meta.yaml`, if any, is not a
    subtype of `cls`.
    """

    config = read_meta(path).config or {}

    written_type = get_scope().get(config.get("type", None), None)

    if path.is_file():
        raise FileExistsError(f"{path} is a file.")

    # if context.enforce_config_match:

    if not path.is_dir():
        if _implements_init(cls) and not get_defaults(cls):
            raise FileNotFoundError(
                f"cannot initialize {path}. It does not yet exist"
                f" and no config was provided to initialize it."
            )
    else:
        pass

    if written_type is not None and not issubclass(written_type, type(cls)):
        raise FileExistsError(
            f"{path} is a {written_type.__module__}.{written_type.__qualname__}"
            f", not a {cls.__module__}.{cls.__qualname__}."
        )

    # if context.enforce_config_match:
    directory = _forward_subclass(type(cls), config)
    # else:
    #     directory = _forward_subclass(type(cls), {})

    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)
    return directory


def _directory_from_config(cls: Directory, conf: Mapping[str, object]) -> Directory:
    """
    Find or build a Directory with the given type and config.
    """
    directory = _forward_subclass(type(cls), conf)
    new_dir_path = _new_directory_path(type(directory))
    object.__setattr__(directory, "_cached_keys", set())
    config = Namespace(**directory._config)

    def _new_directory():
        object.__setattr__(directory, "path", new_dir_path)
        # don't build cause only the type field is populated
        if list(config.keys()) == ["type"]:
            return directory
        # don't build cause the config matches the defaults and init is not implemented
        if not _implements_init(cls) and config.without("type") == get_defaults(cls):
            return directory
        # catches FileExistsError for the case when two processes try to
        # build the same directory simultaneously
        try:
            _build(directory)
        except FileExistsError:
            return _directory_from_config(cls, conf)
        return directory

    for path in Path(get_root_dir()).glob("*"):
        meta = read_meta(path)

        if meta.config == config:
            if getattr(meta, "modified", False):
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        (
                            f"Skipping Directory {path}, which has been modified after "
                            " being build."
                            + "\nYou can use the explicit path as constructor (see "
                            " Directory docs)."
                        ),
                        ModifiedWarning,
                        stacklevel=2,
                    )
                continue

            while meta.status == "running":
                sleep(1.0)
                meta = read_meta(path)
            if meta.status == "done":
                object.__setattr__(directory, "path", path)
                return directory
    return _new_directory()


def _directory_from_path_and_config(
    cls: Directory, path: Path, conf: Mapping[str, object]
) -> Directory:
    """
    Find or build a Directory with the given type, path, and config.
    """
    directory = _forward_subclass(type(cls), conf)
    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)

    if path.exists():
        meta = read_meta(path)
        config = Namespace({"type": _identify(type(directory)), **directory._config})
        if meta.config != config:
            with warnings.catch_warnings():
                if context.enforce_config_match and meta.config is not None:
                    raise FileExistsError(
                        f'"{directory.path}" (incompatible config):\n'
                        f'{config.diff(meta.config, name1="passed", name2="stored")}'
                    )
                elif meta.config is not None:
                    warnings.simplefilter("always")
                    warnings.warn(
                        (
                            f'"{directory.path}" (incompatible config):\n'
                            f'{config.diff(meta.config, name1="passed", name2="stored")}'
                        ),
                        Warning,
                        stacklevel=2,
                    )
        while meta.status == "running":
            sleep(0.01)
            meta = read_meta(path)
        if directory.meta.status == "stopped":
            raise FileExistsError(f'"{directory.path}" was stopped mid-build.')
    else:
        # don't build cause only the type field is populated
        if list(directory._config.keys()) == ["type"]:
            return directory
        # don't build cause the config matches the defaults and init is not implemented
        if not _implements_init(cls) and directory._config.without(
            "type"
        ) == get_defaults(cls):
            return directory
        # catches FileExistsError for the case when two processes try to
        # build the same directory simultaneously
        try:
            _build(directory)
        except FileExistsError:
            return _directory_from_path_and_config(cls, path, conf)
    return directory


def _build(directory: Directory) -> None:
    """
    Create parent directories, invoke `Directory.__init__`, and store metadata.
    """

    if directory.path.exists() and directory.status == "done":
        return
    elif directory.path.exists() and directory.status == "running":
        sleep(0.01)
        return _build(directory)

    directory.path.mkdir(parents=True)

    meta_path = directory.path / "_meta.yaml"
    config = Namespace(**directory._config)

    write_meta(path=meta_path, config=config, status="running")

    try:
        if callable(getattr(type(directory), "__init__", None)):
            n_build_args = directory.__init__.__code__.co_argcount
            # case 1: __init__(self)
            if n_build_args <= 1:
                build_args = []
                build_kwargs = {}
            # case 2: __init__(self, config)
            elif n_build_args == 2 and any(
                [
                    vn in ["config", "conf"]
                    for vn in directory.__init__.__code__.co_varnames
                ]
            ):
                build_args = [directory._config]
                build_kwargs = {}
            # case 3: __init__(self, foo=1, bar=2) to specify defaults and config
            else:
                kwargs = namespacify(get_defaults_from_init(directory))
                assert kwargs
                build_args = []
                build_kwargs = {k: directory._config[k] for k in kwargs}

            # import pdb; pdb.set_trace()

            directory.__init__(*build_args, **build_kwargs)

        write_meta(path=meta_path, config=config, status="done")
    except BaseException as e:
        write_meta(path=meta_path, config=config, status="stopped")
        raise e


def call_signature(cls):
    signature = """

Example call signature:
    {}
    {}
Note, these use the `Directory(config: Dict[str, object])` constructor and are
inferred from defaults and annotations, i.e. they are equivalent to constructing
without arguments."""
    defaults = to_dict(get_defaults(cls))
    string = ""
    for key, val in defaults.items():
        string += f"{key}={val}, "
    if string.endswith(", "):
        string = string[:-2]
    # variant 1: unpacking config kwargs
    signature1 = ""
    if string:
        signature1 = f"{cls.__qualname__}({string})"
    # variant 2: whole config
    signature2 = ""
    if defaults:
        signature2 = f"{cls.__qualname__}({defaults})"

    if signature1 and signature2:
        return signature.format(signature1, signature2)

    return signature.format("(specify defaults for auto-doc of call signature)", "")


def type_signature(cls):
    signature = """

    Types of config elements:
       {}

    """
    annotations = to_dict(get_annotations(cls))

    def qualname(annotation):
        origin = get_origin(annotation)
        if origin:
            return repr(annotation)
        return annotation.__qualname__

    signature1 = ""
    for key, val in annotations.items():
        signature1 += f"{key}: {qualname(val)}, "
    if signature1.endswith(", "):
        signature1 = signature1[:-2]
    if signature1:
        return signature.format(signature1)
    return signature.format("(annotate types for auto-doc of type signature)")


def _auto_doc(cls: type, cls_doc: bool = True, base_doc: bool = False) -> str:
    """Generates automatic documentation string for a Directory class.

    Args:
        cls: Class to document
        cls_doc: Whether to include the class's own docstring
        base_doc: Whether to include base class docstring

    Returns:
        Combined documentation string including call signatures and type info
    """
    docstring = "{}{}{}{}"
    if isinstance(cls, Directory):
        cls = type(cls)
    call_sig = call_signature(cls)
    type_sig = type_signature(cls)

    _cls_doc = ""
    if cls_doc and cls.__doc__:
        _cls_doc = cls.__doc__

    _base_doc = ""
    if base_doc and cls.__base__.__doc:
        _base_doc = cls.__base__.__doc__

    return docstring.format(_cls_doc, call_sig, type_sig, _base_doc)


def _resolve_path(path: Path) -> Path:
    """
    Dereference ".", "..", "~", and "@".
    """
    if path.parts[0] == "@":
        path = get_root_dir() / "/".join(path.parts[1:])
    return path.expanduser().resolve()


def _new_directory_path(type_: type) -> Path:
    """
    Generate an unused path in the Directory root directory.
    """
    # import pdb;pdb.set_trace()
    root = Path(get_root_dir())
    type_name = _identify(type_)
    for i in itertools.count():
        dst = root / f"{type_name}_{i:04x}"
        if not dst.exists():
            return dst.absolute()
    assert False  # for MyPy


def _forward_subclass(cls: type, config: object = {}) -> object:
    """Creates a Directory instance of the appropriate subclass.

    Handles subclass forwarding based on config['type'] and creates instance
    with merged configuration.

    Args:
        cls: Base Directory class
        config: Configuration object or mapping

    Returns:
        New Directory instance of appropriate subclass

    Warns:
        ConfigWarning: When creating dynamic subclass due to unresolved type
    """
    # Coerce `config` to a `dict`.
    config = dict(
        config if isinstance(config, Mapping) else getattr(config, "__dict__", {})
    )

    # Perform subclass forwarding.
    cls_override = config.pop("type", None)
    if isinstance(cls_override, type):
        cls = cls_override
    elif isinstance(cls_override, str):
        try:
            if "." in cls_override:  # hydra-style `type` field
                paths = list(cls_override.split("."))
                cls = import_module(paths[0])
                for path in paths[1:]:
                    cls = getattr(cls, path)
            else:  # legacy scope management
                cls = get_scope()[cls_override]
        except KeyError:
            cls = type(cls_override, (Directory,), {})
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (
                        "Casting to a new subclass of Directory because "
                        f'"{cls_override}" can\'t be resolved as it is not found'
                        + " inside the current scope of Directory subclasses."
                        + " This dynamically created subclass allows to view the data"
                        + " without access to the original class definition and methods."
                        + " If this happens unexpectedly with autoreload enabled in"
                        + " a notebook/IPython session, run `datamate.reset_scope(datamate.Directory)`"
                        + " as a workaround or restart the kernel"
                        + " (background: https://github.com/ipython/ipython/issues/12399)."
                    ),
                    ConfigWarning,
                    stacklevel=2,
                )
            # raise KeyError(
            #     f'"{cls_override}" can\'t be resolved because it is not found'
            #     + f" inside the current scope of Directory subclasses."
            #     + " If this happens unexpectedly with autoreload enabled in"
            #     + " a notebook/IPython session, run `datamate.reset_scope(datamate.Directory)`"
            #     + " as a workaround or restart the kernel"
            #     + f" (background: https://github.com/ipython/ipython/issues/12399)."
            # ) from e

    # Construct and return a Directory instance
    obj = object.__new__(cls)
    default_config = get_defaults(cls)
    default_config.update(config)
    config = Namespace(type=_identify(type(obj)), **default_config)
    object.__setattr__(obj, "_config", namespacify(config))
    return cast(Directory, obj)
