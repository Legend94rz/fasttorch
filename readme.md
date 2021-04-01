# Fast Torch

A keras-like library for pytorch.
Easy to use and more efficient.
In most cases, the only thing you need to do is to define a `nn.Module`,
write the `forward`, and call `Learner(module, optim, loss).fit()` with the help of FastTorch.


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
from fasttorch import *
from torch import nn
import numpy as np


class SimpleMLP(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ln = nn.Sequential(nn.Linear(20, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.ln(x)


if __name__ == "__main__":
    # generate some data
    X = np.random.randn(500000, 20).astype('float32')
    y = (np.median(X, axis=1, keepdims=True)>0).astype('float32')
    print(y.mean())

    # fast torch:
    m = Learner(SimpleMLP(), AdaBelief, BinaryLabelSmoothLoss(0.05))
    m.fit(TensorDataLoader(X[:400000], y[:400000], batch_size=4096, shuffle=True), 1000, None,
          metrics=[(0, 'acc', binary_accuracy_with_logits)],
          callbacks=[EarlyStoppingCallback(verbose=True, patience=7), ReduceLROnPlateauCallback(verbose=True)],
          validation_set=TensorDataLoader(X[400000:], y[400000:], batch_size=4096), verbose=True)
```


## About distributed training

Firstly, the following line should be added before initializing a learner (and the datasets):

`local_rank = Learner.init_distributed_training(dummy=False, seed=0)`

the `dummy` param is used to debug. If user want to disable parallel temporarily, set `dummy=True`.
This function will return the `LOCAL_RANK` mentioned by `torch.distributed.launch` tool. `seed` is the random seed
used by all the training process, which is optional. FastTorch will choose a random value when it is `None` and ensure
all the processes have same random seeds.

Then start parallel training with the help of the tool `torch.distributed.launch` offered by pytorch:

`python -m torch.distributed.launch --use_env [your script and parameters]`

NOTE:
1. `--use_env` is **required** because FastTorch reads the `LOCAL_RANK` from `os.environ`,
   avoiding parses arguments from command line.

1. When using `ModelCheckpoint`,
   users should ensure only one process will save the checkpoint.

   For example, let the process whose `local_rank == 0` writes the checkpoint file:
    ```{python}
    m.fit(train_loader, 100, 256,
          metrics=[(0, 'acc', binary_accuracy_with_logits)],
          callbacks=[ModelCheckpoint('nextshot_{epoch}_{val_acc}.pt', save_best_only=True, verbose=True)] if local_rank==0 else None,
          validation_set=val_loader, verbose=True)
    ```

2. FastTorch will add `DistributedSampler` automatically when the values of `training_set` or `validation_set` is not `torch.DataLoader`.
   Besides, users needn't call `sampler.set_epoch` at every epoch beginning, FastTorch will do that for you.
   
3. Doesn't support distributed training in jupyter notebook now.


## For more complex module

Overwrite `Learner.compute_forward`, `Learner.compute_losses`, `Learner.compute_metric`, and `Learner.compute_output` respectively
to custom the data flow.



# Reference

[1] [torch.distributed.launch.py](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py) and [its tutorial](https://pytorch.org/docs/stable/distributed.html#launch-utility)

