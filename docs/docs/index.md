<h1>
<p style="text-align:left;">
    <img id="datamate-logo-light-content" src="images/datamate_logo_light.webp" width="50%" class="center">
    <img id="datamate-logo-dark-content" src="images/datamate_logo_dark.webp" width="50%" class="center">
</p>
</h1>

## Datamate Documentation

Datamate is a lightweight data and configuration management framework in Python, tailored to support research in machine-learning science. It provides a programming interface to store and retrieve files on a hierarchical filesystem through Directory objects, enabling data creation and access with standard Python code. Built on HDF5 and numpy, it handles file I/O while treating the filesystem as memory.

### Example

The following example demonstrates how to set up an experiment directory, store experiment configuration in _meta.yaml, and store arrays as HDF5 files without boilerplate code. More examples can be found in the [examples](examples/01a_datamate_examples.md) section.

```python
import datamate
import numpy as np

# Set up experiment directory
datamate.set_root_dir("./experiments")

# Set up experiment configuration
config = {
    "model": "01",
    "date": "2024-03-20",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "n_epochs": 100,
    "description": "Setting the learning rate to 0.001"
}

# Set up experiment directory at ./experiments/vision_study/model_01
# and store configuration in _meta.yaml
exp = datamate.Directory("vision_study/model_01", config)

# Store arrays as HDF5 files
exp.images = np.random.rand(100, 64, 64)  # stored as images.h5
exp.responses = np.zeros((100, 1000))     # stored as responses.h5

# Verify that the experiment data is set up
print(exp)

def train(exp: datamate.Directory):
    """Train a model using the experiment's config and data."""
    # Set up optimizer using config
    optimizer = get_optimizer(exp.config.optimizer, lr=exp.config.learning_rate)

	losses = []
    # Training loop using config parameters
    for epoch in range(exp.config.n_epochs):
        # ... training code ...

		# Cache results in memory to avoid high IO overhead
		losses.append(loss)

	# Store results back in experiment directory outside of the training loop
	exp.losses = np.array(losses)  # creates losses.h5

# Run training
train(exp)

# Access results
mean_loss = exp.losses.mean()  # compute mean loss
```

### Main Features

The main features of `datamate` are:

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
- Basic experiment status tracking


### Installation

```pip install datamate```

### Tutorials
- [datamate Examples](examples/01a_datamate_examples.md)
- [Parallel read/write operations](examples/01b_parallel_read_and_write.md)

### API Reference

For detailed information about Datamate's components and functions, please refer to our [API Reference](reference/directory.md) section.

### Related Projects

`datamate` was co-developed with the [flyvis](https://github.com/turagalab/flyvis) project, which is a complex usage example of `datamate`.

[artisan](https://github.com/MasonMcGill/artisan) is the original framework that inspired `datamate`.

### Getting Help

If you have any questions or encounter any issues, please check our [FAQ](faq.md) or [Contributing](contribute.md) pages for more information.
