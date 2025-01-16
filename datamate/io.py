"""Module for handling file I/O operations for Directory objects.

This module provides functionality for reading, writing, and manipulating HDF5 files
and directories.
"""

import shutil
from pathlib import Path
from typing import (
    Any,
    Protocol,
    Tuple,
    runtime_checkable,
    TYPE_CHECKING,
    Optional,
    Dict,
)
from time import sleep
import h5py as h5
import numpy as np
from pandas import DataFrame

# Only import Directory for type checking
if TYPE_CHECKING:
    from .directory import Directory

__all__ = [
    "ArrayFile",
    "H5Reader",
    "_read_h5",
    "_write_h5",
    "_extend_h5",
    "_copy_file",
    "_copy_dir",
    "_extend_file",
    "directory_to_dict",
    "directory_to_df",
]


@runtime_checkable
class ArrayFile(Protocol):
    """Protocol for single-array HDF5 file interface.

    Attributes:
        path: Path to the HDF5 file.
        shape: Shape of the array data.
        dtype: NumPy dtype of the array data.
    """

    path: Path
    shape: Tuple[int, ...]
    dtype: np.dtype

    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...
    def __getattr__(self, key: str) -> Any: ...


# -- I/O -----------------------------------------------------------------------


class H5Reader(ArrayFile):
    """Wrapper around HDF5 read operations with safe file handle management.

    Ensures file handles are only open during access operations to prevent resource leaks.

    Attributes:
        path: Path to the HDF5 file.
        shape: Shape of the array data.
        dtype: NumPy dtype of the array data.
        n_retries: Number of retry attempts for file operations.
    """

    def __init__(
        self, path: Path, assert_swmr: bool = True, n_retries: int = 10
    ) -> None:
        self.path = Path(path)
        with h5.File(self.path, mode="r", libver="latest", swmr=True) as f:
            if assert_swmr:
                assert f.swmr_mode, "File is not in SWMR mode."
            assert "data" in f
            self.shape = f["data"].shape
            self.dtype = f["data"].dtype
        self.n_retries = n_retries

    def __getitem__(self, key):
        for retry_count in range(self.n_retries):
            try:
                with h5.File(self.path, mode="r", libver="latest", swmr=True) as f:
                    data = f["data"][key]
                break
            except Exception as e:
                if retry_count == self.n_retries - 1:
                    raise e
                sleep(0.1)
        return data

    def __len__(self):
        return self.shape[0]

    def __getattr__(self, key):
        # get attribute from underlying h5.Dataset object
        for retry_count in range(self.n_retries):
            try:
                with h5.File(self.path, mode="r", libver="latest", swmr=True) as f:
                    value = getattr(f["data"], key, None)
                break
            except Exception as e:
                if retry_count == self.n_retries - 1:
                    raise e
                sleep(0.1)
        if value is None:
            raise AttributeError(f"Attribute {key} not found.")
        # wrap callable attributes to open file before calling function
        if callable(value):

            def safe_wrapper(*args, **kwargs):
                # not trying `n_retries` times here, just for simplicity
                with h5.File(self.path, mode="r", libver="latest", swmr=True) as f:
                    output = getattr(f["data"], key)(*args, **kwargs)
                return output

            return safe_wrapper
        # otherwise just return value
        else:
            return value


def _read_h5(path: Path, assert_swmr: bool = True) -> ArrayFile:
    """Read HDF5 file with retry mechanism.

    Args:
        path: Path to the HDF5 file.
        assert_swmr: Whether to assert Single-Writer-Multiple-Reader mode.

    Returns:
        An ArrayFile interface to the HDF5 file.

    Raises:
        OSError: If file cannot be opened after retries.
    """
    try:
        return H5Reader(path, assert_swmr=assert_swmr)
    except OSError as e:
        print(f"{path}: {e}")
        if "errno = 2" in str(e):
            raise e
        sleep(0.1)
        return _read_h5(path)


def _write_h5(path: Path, val: np.ndarray) -> None:
    """Write array data to HDF5 file.

    Args:
        path: Path to the HDF5 file.
        val: Array data to write.
    """
    val = np.asarray(val)
    try:
        f = h5.File(path, libver="latest", mode="w")
        if f["data"].dtype != val.dtype:
            raise ValueError()
        f["data"][...] = val
        f.swmr_mode = True
        assert f.swmr_mode
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        f = h5.File(path, libver="latest", mode="w")
        f["data"] = val
        f.swmr_mode = True
        assert f.swmr_mode
    f.close()


def _extend_h5(path: Path, val: object, retry: int = 0, max_retries: int = 50) -> None:
    val = np.asarray(val)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mode='a' to read file, create otherwise
    try:
        f = h5.File(path, libver="latest", mode="a")
        if "data" not in f:
            dset = f.require_dataset(
                name="data",
                shape=None,
                maxshape=(None, *val.shape[1:]),
                dtype=val.dtype,
                data=np.empty((0, *val.shape[1:]), val.dtype),
                chunks=(
                    int(np.ceil(2**12 / np.prod(val.shape[1:]))),
                    *val.shape[1:],
                ),
            )
            f.swmr_mode = True
        else:
            dset = f["data"]
    except BlockingIOError as e:
        print(e)
        if "errno = 11" in str(e) or "errno = 35" in str(
            e
        ):  # 11, 35 := Reource temporarily unavailable
            sleep(0.1)
            if retry < max_retries:
                _extend_h5(path, val, retry + 1, max_retries)
            else:
                raise RecursionError(
                    "maximum retries to extend the dataset"
                    " exceeded, while the resource was unavailable. Because"
                    " the dataset is constantly locked by another thread."
                )
            return
        else:
            raise e

    def _override_to_chunked(path: Path, val: object) -> None:
        # override as chunked dataset
        data = _read_h5(path, assert_swmr=False)[()]
        path.unlink()
        _extend_h5(path, data)
        # call extend again with new value
        _extend_h5(path, val)

    if len(val) > 0:
        try:
            dset.resize(dset.len() + len(val), 0)
            dset[-len(val) :] = val
            dset.flush()
        except TypeError as e:
            # workaround if dataset was first created as non-chunked
            # using __setitem__ and then extended using extend
            if "Only chunked datasets can be resized" in str(e):
                _override_to_chunked(path, val)
            else:
                raise e
    f.close()


def _copy_file(dst: Path, src: Path) -> None:
    # shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _copy_dir(dst: Path, src: Path) -> None:
    """Copy a directory and its metadata.

    Args:
        dst: Destination path
        src: Source directory path
    """
    # Create parent directory if needed
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy directory contents
    shutil.copytree(src, dst)

    # Ensure metadata is copied correctly
    meta_src = src / "_meta.yaml"
    meta_dst = dst / "_meta.yaml"
    if meta_src.exists():
        shutil.copy2(meta_src, meta_dst)


def _copy_if_conflict(src):
    _existing_conflicts = src.parent.glob(f"{src.name}_conflict*")
    max_count = max([int(c.name[-4:]) for c in _existing_conflicts])
    dst = src.parent / f"{src.name}_conflict_{max_count+1:04}"
    shutil.copytree(src, dst)
    return dst


def _extend_file(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_src:
        with open(dst, "ab+") as f_dst:
            f_dst.write(f_src.read())


def directory_to_dict(directory: "Directory") -> dict:
    dw_dict = {
        key: getattr(directory, key)[...]
        for key in list(directory.keys())
        if isinstance(getattr(directory, key), H5Reader)
    }
    return dw_dict


def directory_to_df(
    directory: "Directory", dtypes: Optional[Dict[str, np.dtype]] = None
) -> DataFrame:
    """Convert a directory to a pandas DataFrame.

    Creates a DataFrame from HDF5 datasets in the directory. Single-element datasets
    are broadcast to match the most common length.

    Args:
        directory: Directory object containing HDF5 datasets.
        dtypes: Optional mapping of column names to NumPy dtypes.

    Returns:
        DataFrame containing the directory data.

    Example:
        ```python
        dir = Directory("path/to/dir")
        df = directory_to_df(dir, {"col1": np.float32})
        ```
    """
    from .utils import byte_to_str

    df_dict = {
        key: getattr(directory, key)[...]
        for key in list(directory.keys())
        if isinstance(getattr(directory, key), H5Reader)
    }

    # Get the lengths of all datasets.
    nelements = {k: len(v) or 1 for k, v in df_dict.items()}

    lengths, counts = np.unique([val for val in nelements.values()], return_counts=True)
    most_frequent_length = lengths[np.argmax(counts)]

    # If there are single element datasets, just create a new column of most_frequent_length and put the value in each row.
    if lengths.min() == 1:
        for k, v in nelements.items():
            if v == 1:
                df_dict[k] = df_dict[k].repeat(most_frequent_length)

    df_dict = byte_to_str(df_dict)

    if dtypes is not None:
        df_dict = {
            k: np.array(v).astype(dtypes[k]) for k, v in df_dict.items() if k in dtypes
        }
    return DataFrame.from_dict(
        {k: v.tolist() if v.ndim > 1 else v for k, v in df_dict.items()}
    )
