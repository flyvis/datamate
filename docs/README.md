# Documentation

The documentation is available at: <https://flyvis.github.io/datamate/>

## Building the Documentation

Run the [build docs script](build_docs.sh) or follow the instructions below to build the documentation.

The documentation is built with [mkdocs](https://www.mkdocs.org/).

### Convert examples to markdown:

Only applies to making changes to the examples.

1. Run all notebooks inplace:

```bash
export JUPYTER_CONFIG_DIR=$(mktemp -d) # to avoid conflicts with notebook version and extensions

for notebook in ../examples/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$notebook" --inplace
done
```

2. Convert notebooks to markdown:

```bash
jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir docs/examples/ --TagRemovePreprocessor.remove_cell_tags hide
```

3. Clear all notebook outputs (optional):
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ../examples/*.ipynb
```

## Serve the docs locally

```bash
mkdocs serve
```

## Deploy the docs to GitHub

See [mkdocs user guide](https://www.mkdocs.org/user-guide/deploying-your-docs/) for more details.

```bash
mkdocs gh-deploy
```

or optionally specify a remote repository:

```bash
mkdocs gh-deploy --remote <remote_name>
```
