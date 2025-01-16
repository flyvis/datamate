"""Module for handling metadata reading, writing and validation for Directory objects."""

from pathlib import Path
from typing import Dict, Any, Optional, Literal
import warnings
import numpy as np
from time import sleep
from ruamel.yaml import YAML

from .namespaces import Namespace, namespacify


__all__ = [
    "read_meta",
    "write_meta",
    "MetadataError",
    "MetadataParseError",
    "MetadataValidationError",
]


class MetadataError(ValueError):
    """Base class for metadata-related errors."""

    pass


class MetadataParseError(MetadataError):
    """Raised when metadata YAML cannot be parsed."""

    pass


class MetadataValidationError(MetadataError):
    """Raised when metadata structure is invalid."""

    pass


def read_meta(path: Path, retries: int = 5) -> Namespace:
    """Read and validate metadata from a directory's `_meta.yaml` file.

    Args:
        path: Directory path containing `_meta.yaml`
        retries: Number of retry attempts for transient failures

    Returns:
        Namespace containing validated metadata with `config` and `status`

    Raises:
        MetadataParseError: If YAML parsing fails
        MetadataValidationError: If metadata structure is invalid
        FileNotFoundError: If `_meta.yaml` doesn't exist
        NotADirectoryError: If path is not a directory

    Note:
        Returns default metadata (empty config, status="done") for non-existent paths
    """
    meta_path = path / "_meta.yaml"

    try:
        yaml = YAML()
        with open(meta_path, "r") as f:
            try:
                meta = yaml.load(f)
            except Exception as e:
                raise MetadataParseError(f"Failed to parse {meta_path}: {e}") from e

        meta = namespacify(meta)

        # Validate metadata structure
        if not isinstance(meta, Namespace):
            raise MetadataValidationError(
                f"Metadata must be a Namespace, got {type(meta)}"
            )

        if not hasattr(meta, "config"):
            raise MetadataValidationError(
                f"Missing required 'config' field in {meta_path}"
            )

        if not isinstance(meta.config, Namespace):
            raise MetadataValidationError(
                f"'config' must be a Namespace in {meta_path}"
            )

        if not hasattr(meta, "status"):
            raise MetadataValidationError(
                f"Missing required 'status' field in {meta_path}"
            )

        if not isinstance(meta.status, str):
            raise MetadataValidationError(f"'status' must be a string in {meta_path}")

        # Handle legacy 'spec' field
        if hasattr(meta, "spec"):
            if not isinstance(meta.spec, Namespace):
                raise MetadataValidationError(
                    f"Legacy 'spec' must be a Namespace in {meta_path}"
                )
            warnings.warn(
                f"Directory {path} uses legacy 'spec' instead of 'meta'. Please update.",
                DeprecationWarning,
                stacklevel=2,
            )
            meta["config"] = meta.pop("spec")

        return meta

    except (MetadataError, AssertionError) as e:
        if retries > 0:
            sleep(0.1)
            return read_meta(path, retries=retries - 1)
        raise e

    except (FileNotFoundError, NotADirectoryError):
        # Return default metadata for non-existent or invalid paths
        return Namespace(config=None, status="done")


def write_meta(
    path: Path,
    config: Optional[Dict[str, Any]] = None,
    status: Optional[Literal["done", "error", "running"]] = None,
    **kwargs: Any,
) -> None:
    """Write metadata to a YAML file.

    Args:
        path: Path to write the metadata file
        config: Configuration dictionary to store
        status: Status string indicating directory state
        **kwargs: Additional metadata fields to include

    Note:
        Supports serialization of numpy types (arrays, integers, floats)
    """
    yaml = YAML()

    # support dumping numpy objects
    def represent_numpy_float(self, value):
        return self.represent_float(float(value))

    def represent_numpy_int(self, value):
        return self.represent_int(int(value))

    def represent_numpy_array(self, value):
        return self.represent_sequence(value.tolist())

    yaml.Representer.add_multi_representer(np.ndarray, represent_numpy_array)
    yaml.Representer.add_multi_representer(np.floating, represent_numpy_float)
    yaml.Representer.add_multi_representer(np.integer, represent_numpy_int)

    # This allows directory types to be dumped to yaml
    config = _identify_elements(config)
    kwargs = _identify_elements(kwargs)

    # dump config to yaml
    with open(path, "w") as f:
        yaml.dump({"config": config, "status": status, **kwargs}, f)


def _identify_elements(obj: Any) -> Any:
    """Recursively identify elements in config objects for YAML serialization.

    Args:
        obj: Object to process for serialization

    Returns:
        Processed object ready for YAML serialization
    """
    if isinstance(obj, type):
        return _identify(obj)
    elif isinstance(obj, list):
        return [_identify_elements(elem) for elem in obj]
    elif isinstance(obj, dict):
        return {k: _identify_elements(obj[k]) for k in obj}
    else:
        return obj


def _identify(type_: type) -> str:
    """Convert a type to its string identifier in the current scope.

    Args:
        type_: Type to identify

    Returns:
        String identifier for the type
    """
    from .directory import get_scope

    for sym, t in get_scope().items():
        if t.__qualname__ == type_.__qualname__:
            return sym
