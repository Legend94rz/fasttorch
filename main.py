from fasttorch import EarlyStoppingCallback, ReduceLROnPlateauCallback, Learner, binary_accuracy_with_logits, TensorDataLoader, AdaBelief
from torch import nn
import torch as T
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
    m = Learner(SimpleMLP())
    m.fit(TensorDataLoader(X[:400000], y[:400000], batch_size=4096, shuffle=True), 1000, None, AdaBelief, T.nn.functional.binary_cross_entropy_with_logits,
          metrics=[(0, 'acc', binary_accuracy_with_logits)],
          callbacks=[EarlyStoppingCallback(verbose=True, patience=7), ReduceLROnPlateauCallback(verbose=True)],
          validation_set=TensorDataLoader(X[400000:], y[400000:], batch_size=4096), verbose=True)
