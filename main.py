from fasttorch import EarlyStoppingCallback, ReduceLROnPlateauCallback, Learner, binary_accuracy_with_logits, ModelCheckpoint
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
    X = np.random.randn(50000, 100).astype('float32')
    X[:, 0] = np.random.randint(0, 2, (50000, )).astype('float32')
    y = X[:, 0].reshape(-1, 1)
    m = Learner(SimpleMLP())
    m.fit((X[:40000], y[:40000]), 100, 256, T.optim.Adam, T.nn.functional.binary_cross_entropy_with_logits,
          metrics=[(0, 'acc', binary_accuracy_with_logits)],
          callbacks=[EarlyStoppingCallback(verbose=True), ReduceLROnPlateauCallback(verbose=True),
                     ModelCheckpoint('learner.pt', verbose=True, save_best_only=True)],
          validation_set=(X[40000:], y[40000:]))

