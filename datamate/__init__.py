"""A data organization and compilation system.

This package provides tools for organizing and managing data directories with a focus on
array-based file handling and directory management.
"""

from datamate.directory import (
    Directory,
    DirectoryDiff,
    ArrayFile,
)

from datamate.context import (
    set_root_dir,
    get_root_dir,
    enforce_config_match,
    check_size_on_init,
    get_check_size_on_init,
    set_verbosity_level,
    root,
    set_root_context,
    reset_scope,
    delete_if_exists,
    context,
)

from datamate.namespaces import Namespace, namespacify
from datamate.version import __version__

__all__ = [
    "ArrayFile",
    "Directory",
    "DirectoryDiff",
    "Namespace",
    "namespacify",
    "set_root_dir",
    "get_root_dir",
    "enforce_config_match",
    "check_size_on_init",
    "get_check_size_on_init",
    "set_verbosity_level",
    "root",
    "set_root_context",
    "reset_scope",
    "delete_if_exists",
    "context",
    "__version__",
]
