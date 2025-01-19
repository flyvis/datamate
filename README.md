# Datamate

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
