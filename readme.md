# Fast Torch

A keras-like library for pytorch.
Easy to use and more efficient.
With the help of FastTorch, the only thing you should do is to define a `nn.Module`, 
write the `forward`, and call `Learner(module).fit()`. 


# Setup

1. clone this repo:

    `git clone https://github.com/Legend94rz/fasttorch`


2. setup by `setup.py`:

    `python setup.py install`

    or, you can build a `*.whl` package and then install it by `pip`:

    ```
    python setup.py bdist_wheel
    pip install -U (the-whl-file-name-generated-just-now).whl
    ```

# Tutorial
## Example code

```python
from fasttorch import EarlyStoppingCallback, ReduceLROnPlateauCallback, Learner, binary_accuracy_with_logits, TensorDataLoader
from torch import nn
import torch as T
import numpy as np

# Define your model:
class SimpleMLP(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ln = nn.Linear(100, 1)

    def forward(self, x):
        return self.ln(x)


if __name__ == "__main__":
    # generate some data
    X = np.random.randn(50000, 100).astype('float32')
    X[:, 0] = np.random.randint(0, 2, (50000, )).astype('float32')
    y = X[:, 0].reshape(-1, 1)

    # fast torch:
    m = Learner(SimpleMLP())
    m.fit(TensorDataLoader(X[:40000], y[:40000], batch_size=256), 100, 256, T.optim.Adam, T.nn.functional.binary_cross_entropy_with_logits,
          metrics=[(0, 'acc', binary_accuracy_with_logits)],
          callbacks=[EarlyStoppingCallback(verbose=True), ReduceLROnPlateauCallback(verbose=True)],
          validation_set=(X[40000:], y[40000:]), verbose=True)
```


## About distributed training

Firstly, the following line should be added before the initializing of learner (and the datasets):

`local_rank = Learner.init_distributed_training(dummy=False)`

the `dummy` param is used to debug. If user want to disable parallel temporarily, set `dummy=True`.
This function will return the `LOCAL_RANK` mentioned by `torch.distributed.launch` tool.

Then start parallel training with the help of the tool `torch.distributed.launch` offered by pytorch:

`python -m torch.distributed.launch [your script].py`

NOTE:
1. When using `ModelCheckpoint`, 
users should ensure only the process whose local rank (or rank in global) equals to 0 saves the checkpoint.

    For example:
    ```
    m.fit(train_loder, 100, 256, T.optim.Adam, T.nn.functional.binary_cross_entropy_with_logits,
          metrics=[(0, 'acc', binary_accuracy_with_logits)],
          callbacks=[ModelCheckpoint('nextshot_{epoch}_{val_acc}.pt', save_best_only=True, verbose=True)] if local_rank==0 else None,
          validation_set=val_loader, verbose=True)
    ```

2. Under distributed training scenario, the params `training_set` and `validation_set` in the `fit` function only support offical `DataLoader` instance now.
Ensure they have set `sampler` properly.
Users needn't call `sampler.set_epoch` at every epoch beginning, FastTorch will do that for you.


## For more complex module

Overwrite `Learner.forward`, `Learner.compute_loss`, and `Learner.compute_metric` respectively
to custom the data flow.



# Reference

[1] [torch.distributed.launch.py](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py) and [its tutorial](https://pytorch.org/docs/stable/distributed.html#launch-utility)

