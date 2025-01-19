## Parallel read/write operations

This example requires `01a_datamate_intro.ipynb` to be run simultaneously.

We monitor the `Directory` called "NetworkDir_0000" from `01a_datamate_into` to track the training progress.


```python
from pathlib import Path
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

import datamate

root_dir = Path(".") / "data"
datamate.set_root_dir(root_dir)
```


```python
# we instantiate a pointer to the known Directory
network_dir = datamate.Directory("NetworkDir_0000")
```

    /Users/janne/projects/datamate/datamate/directory.py:1352: ConfigWarning: Casting to a new subclass of Directory because "NetworkDir" can't be resolved as it is not found inside the current scope of Directory subclasses. This dynamically created subclass allows to view the data without access to the original class definition and methods. If this happens unexpectedly with autoreload enabled in a notebook/IPython session, run `datamate.reset_scope(datamate.Directory)` as a workaround or restart the kernel (background: https://github.com/ipython/ipython/issues/12399).
      directory = _forward_subclass(type(cls), config)



```python
network_dir.meta
```




    Namespace(
      config = Namespace(type='NetworkDir', tau=200.0, sigma=0.1, num_iters=100),
      status = 'done'
    )




```python
# we visualize the loss to monitor the training


def watch_loss(network_dir, updates=100):
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(network_dir.loss[:])
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")

    def update_loss(loss):
        iters = np.arange(0, len(loss))
        ax.lines[0].set_data(iters, loss)
        print(f"Current loss: {loss[-1]:.2f}", end="\r")
        if loss.any():
            ymax = np.max(loss)
            ymin = np.min(loss)
        ax.axis([0, iters[-1], ymin, ymax])

    while network_dir.meta.status == "running":
        loss = network_dir.loss[:]
        update_loss(loss)
        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(0.1)
        updates -= 1
```


```python
%matplotlib notebook
watch_loss(network_dir)
```


    <IPython.core.display.Javascript object>



<div id='068051ab-c476-4a47-9bd1-9cbaff0ab5ed'></div>



```python

```
