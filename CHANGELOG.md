# Changelog

## [1.0.0] - 2025-01-23

### Breaking
- Reorganized package structure into separate modules for better maintainability
  - Moved context management to `context.py`
  - Moved directory comparison to `diff.py`
  - Moved I/O operations to `io.py`
  - Moved metadata handling to `metadata.py`

### Documentation
- Added comprehensive documentation site using MkDocs Material
- Added detailed API Reference
- Updated examples
- Added contributing guidelines and release process
- Added project logo and branding

### Infrastructure
- Migrated from black to ruff for code formatting and linting
- Added pre-commit hooks for code quality
- Fixed GitHub Actions workflow for automated testing
- Updated build system with pyproject.toml

### Distribution
- Added proper package metadata and classifiers
- Added development and documentation dependencies
