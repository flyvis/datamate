<h1>
<p style="text-align:left;">
    <img src="docs/docs/images/datamate_logo_light.webp" width="50%" alt="Datamate Logo">
</p>
</h1>

[![PyPI version](https://badge.fury.io/py/datamate.svg)](https://badge.fury.io/py/datamate)
[![Python](https://img.shields.io/pypi/pyversions/datamate.svg)](https://pypi.org/project/datamate/)
[![Tests](https://github.com/flyvis/datamate/actions/workflows/auto_test.yml/badge.svg)](https://github.com/flyvis/datamate/actions/workflows/auto_test.yml)
[![codecov](https://codecov.io/gh/flyvis/datamate/branch/main/graph/badge.svg)](https://codecov.io/gh/flyvis/datamate)
[![License](https://img.shields.io/github/license/flyvis/datamate.svg)](https://github.com/flyvis/datamate/blob/main/LICENSE)

Datamate is a data and configuration management framework in Python for machine-learning research. It uses the filesystem as memory through Directory objects, providing a programming interface to store and retrieve files in hierarchical structures using HDF5.

## Main Features

- Filesystem as memory through Directory objects
- Hierarchical data organization
- Automatic path handling and resolution with pathlib
- Array storage in HDF5 format
- Parallel read/write operations
- Configuration-based compilation and access of data
- Configuration management in YAML files
- Configuration comparison and diffing
- Pandas DataFrame integration
- Directory structure visualization (tree view)
- Experiment status tracking

## Example

```python
import datamate
import numpy as np

# Set up experiment directory
datamate.set_root_dir("./experiments")

# Set up experiment configuration
config = {
    "model": "01",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "n_epochs": 100
}

# Set up experiment directory and store configuration
exp = datamate.Directory("vision_study/model_01", config)

# Store arrays as HDF5 files
exp.images = np.random.rand(100, 64, 64)  # stored as images.h5
exp.responses = np.zeros((100, 1000))     # stored as responses.h5

# Access data
mean_response = exp.responses[:].mean()
```

More detailed examples in the [documentation](https://flyvis.github.io/datamate).

## Installation

Using pip:

```bash
pip install datamate
```

## Documentation

Full documentation is available at [flyvis.github.io/datamate](https://flyvis.github.io/datamate).

## Related Projects

- [flyvis](https://github.com/turagalab/flyvis) - Usage example of datamate
- [artisan](https://github.com/MasonMcGill/artisan) - The framework that inspired datamate

## Contributing

Contributions welcome! Please check our [Contributing Guide](https://flyvis.github.io/datamate/contribute) for guidelines.
