# Fast Torch

A keras-like library for pytorch.
Easy to use and more efficient.


# Example code

```python
from fasttorch import EarlyStoppingCallback, ReduceLROnPlateauCallback, Learner, binary_accuracy_with_logits, TensorDataLoader
from torch import nn
import torch as T
import numpy as np


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
