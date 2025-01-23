# Release Process

## Prerequisites

1. Ensure all tests pass:
```bash
pytest
```

2. Update and verify the documentation:
Follow the instructions in the [readme](README.md) to update and verify the documentation.
The deployment to github can be done last or via workflow.

3. Install required tools:
```bash
python -m pip install build twine
```

## Release Steps

0. **Test PyPi before committing (optional)**

One can create the wheel and upload it to Test PyPi before committing to develop the package.
This can be useful to test if the installation will work before commiting to a version number and
push to the remote repository. However, the wheels that are uploaded to Test PyPi cannot
be deleted so one should probably do this sparingly and not use version numbers one wants to reserve for
the actual release.

### Upload to Test PyPi
```bash
# Clean previous builds
rm -rf dist/

# Set version temporarily for this session manually (change version number)
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev1

# Now build and test
python -m build

# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*
```

### Test Installation

In a clean environment, run these sporadic tests to verify the installation:
```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "datamate==0.0.0.dev1"

python -c "from datamate import Directory; d = Directory(); assert not d.path.exists()"

pip uninstall datamate -y

unset SETUPTOOLS_SCM_PRETEND_VERSION
```

### Commit Changes

Commit all open changes to the repository.

### Update Changelog

- Append entry in `CHANGELOG.md` with new version number
- Include all notable changes under appropriate sections, e.g.,
   - Breaking
   - Features
   - Documentation
   - Infrastructure
   - Distribution
   - Bug Fixes

```bash
git add CHANGELOG.md
git commit -m "docs: add changelog for v1.0.0"
```

### Create and Push Tag

```bash
# Create annotated tag using changelog
git tag -a v1.0.0 -F CHANGELOG.md

# Push to both remotes
git push origin main
git push origin v1.0.0
git push public_repo main
git push public_repo v1.0.0
```

### Build and Upload to PyPI
```bash
# Clean previous builds
rm -rf dist/

# Set version temporarily for this session manually
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

### Create GitHub Release
   - Go to GitHub releases page
   - Create new release using the tag
   - Copy changelog entry into release description

## Post-release

1. Verify package can be installed from PyPI:
```bash
python -m pip install datamate
```

## Check documentation is updated on the documentation website

## Version Numbering

We follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

## Notes

- Always test on Test PyPI before releasing to PyPI
