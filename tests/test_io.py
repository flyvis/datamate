"""Tests for datamate.io, covering the h5py open-file error fixes."""

import h5py
import numpy as np
import pytest

from datamate.io import H5Reader, _read_h5, _write_h5

_SWMR_OPEN_ERROR = "Unable to synchronously open file (errno = 2)"


# -- _write_h5 -----------------------------------------------------------------


def test_write_h5_creates_file(tmp_path):
    """_write_h5 must create a valid SWMR HDF5 file."""
    path = tmp_path / "data.h5"
    val = np.array([1.0, 2.0, 3.0])
    _write_h5(path, val)
    assert path.exists()
    with h5py.File(path, mode="r", libver="latest", swmr=True) as f:
        assert "data" in f
        np.testing.assert_array_equal(f["data"][...], val)


def test_write_h5_overwrites_existing_file(tmp_path):
    """_write_h5 must overwrite an existing file without leaving it locked."""
    path = tmp_path / "data.h5"
    val1 = np.array([1.0, 2.0])
    val2 = np.array([9.0, 8.0])
    _write_h5(path, val1)
    _write_h5(path, val2)  # must not raise PermissionError on any platform
    with h5py.File(path, mode="r", libver="latest", swmr=True) as f:
        np.testing.assert_array_equal(f["data"][...], val2)


def test_write_h5_creates_parent_dirs(tmp_path):
    """_write_h5 must create missing parent directories."""
    path = tmp_path / "a" / "b" / "c" / "data.h5"
    _write_h5(path, np.array([42]))
    assert path.exists()


def test_write_h5_replaces_empty_dir_at_path(tmp_path):
    """_write_h5 must replace an empty directory that sits at the target path."""
    path = tmp_path / "data.h5"
    path.mkdir()
    _write_h5(path, np.array([1, 2, 3]))
    assert path.is_file()


# -- H5Reader / _read_h5 -------------------------------------------------------


def test_h5reader_reads_swmr_file(tmp_path):
    """H5Reader must read a file written with SWMR mode."""
    path = tmp_path / "data.h5"
    val = np.array([10, 20, 30])
    _write_h5(path, val)
    reader = H5Reader(path, assert_swmr=True)
    np.testing.assert_array_equal(reader[:], val)
    assert reader.shape == val.shape
    assert reader.dtype == val.dtype


def test_h5reader_fallback_without_swmr(tmp_path):
    """H5Reader must read a non-SWMR file when assert_swmr=False.

    On systems where h5py can open non-SWMR files with swmr=True this test
    simply verifies the read succeeds.  On systems where that raises an
    OSError the fallback path in H5Reader is exercised instead."""
    path = tmp_path / "legacy.h5"
    val = np.array([1.0, 2.0, 3.0])
    # Write without SWMR mode (simulates downloaded files)
    with h5py.File(path, mode="w") as f:
        f["data"] = val

    # Must not raise even though the file has no SWMR metadata
    reader = H5Reader(path, assert_swmr=False)
    np.testing.assert_array_equal(reader[:], val)


def test_h5reader_fallback_activates_on_oserror(tmp_path, monkeypatch):
    """H5Reader must fall back to non-SWMR when h5.File raises OSError."""
    path = tmp_path / "swmr.h5"
    val = np.array([1.0, 2.0, 3.0])
    _write_h5(path, val)

    original_file = h5py.File
    call_count = [0]

    def patched_file(p, mode="r", **kwargs):
        if kwargs.get("swmr") and call_count[0] == 0:
            call_count[0] += 1
            raise OSError(_SWMR_OPEN_ERROR)
        return original_file(p, mode=mode, **kwargs)

    monkeypatch.setattr(h5py, "File", patched_file)
    import datamate.io as io_mod
    monkeypatch.setattr(io_mod, "h5", h5py)

    reader = H5Reader(path, assert_swmr=False)
    # After OSError fallback, _swmr must be False
    assert reader._swmr is False
    np.testing.assert_array_equal(reader[:], val)


def test_h5reader_assert_swmr_raises_for_non_swmr_file(tmp_path, monkeypatch):
    """H5Reader must propagate OSError when assert_swmr=True."""
    path = tmp_path / "non_swmr.h5"
    val = np.array([1, 2, 3])
    with h5py.File(path, mode="w") as f:
        f["data"] = val

    original_file = h5py.File

    def patched_file(p, mode="r", **kwargs):
        if kwargs.get("swmr"):
            raise OSError(_SWMR_OPEN_ERROR)
        return original_file(p, mode=mode, **kwargs)

    monkeypatch.setattr(h5py, "File", patched_file)
    import datamate.io as io_mod
    monkeypatch.setattr(io_mod, "h5", h5py)

    with pytest.raises(OSError):
        H5Reader(path, assert_swmr=True)


def test_h5reader_fallback_chains_exception_on_double_failure(tmp_path, monkeypatch):
    """When both the SWMR open and the fallback open fail, the fallback OSError
    must be chained from the original SWMR OSError."""
    path = tmp_path / "swmr.h5"
    val = np.array([1.0])
    _write_h5(path, val)

    original_file = h5py.File
    call_count = [0]

    def patched_file(p, mode="r", **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise OSError(_SWMR_OPEN_ERROR)
        raise OSError("fallback also failed")

    monkeypatch.setattr(h5py, "File", patched_file)
    import datamate.io as io_mod
    monkeypatch.setattr(io_mod, "h5", h5py)

    with pytest.raises(OSError, match="fallback also failed") as exc_info:
        H5Reader(path, assert_swmr=False)
    assert exc_info.value.__cause__ is not None
    assert _SWMR_OPEN_ERROR in str(exc_info.value.__cause__)


def test_read_h5_fallback_without_swmr(tmp_path):
    """_read_h5 with assert_swmr=False must read files not in SWMR mode."""
    path = tmp_path / "legacy.h5"
    val = np.array([5, 6, 7])
    with h5py.File(path, mode="w") as f:
        f["data"] = val
    reader = _read_h5(path, assert_swmr=False)
    np.testing.assert_array_equal(reader[:], val)
