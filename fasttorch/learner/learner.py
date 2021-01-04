import torch as T
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import sys
from typing import Iterable
import numpy as np
from collections import defaultdict
import pandas as pd
from ..callbacks import CallbackList


class Learner:
    def __init__(self, module):
        self.module = module
        self.stop_training = False

    def _splitor(self, batch, n_target, device):
        input = [c.to(device) for c in batch[:-n_target]]
        target = [c.to(device) for c in batch[-n_target:]]
        return input, target

    def _make_dataloader(self, dataset, batch_size):
        if dataset is None:
            return None
        if isinstance(dataset, Dataset):
            return DataLoader(dataset, batch_size=batch_size)
        elif isinstance(dataset, DataLoader):
            return dataset
        elif isinstance(dataset, np.ndarray) or isinstance(dataset, T.Tensor):
            return DataLoader(TensorDataset(T.tensor(dataset)), batch_size=batch_size)
        elif isinstance(dataset, Iterable):
            dataset = [T.tensor(a) for a in dataset]
            return DataLoader(TensorDataset(*dataset), batch_size=batch_size)
        else:
            raise NotImplementedError

    def fit(self, training_set, epochs, batch_size, optimizer_fn, loss_fn, metrics=None, train_scoring=True, validation_set=None, callbacks=None, device='cpu'):
        """
        training_set: (x1, x2, ..., y1, y2, ...), torch Dataset or DataLoader
        optimizer_fn: callable or optim instance
        loss_fn: callable (including loss instance) or list of these two type for multi target.
                if loss_fn is a list, the last `len(loss_fn)` components of training_set will be considered as labels respectivelly.
                besides, `len(loss_fn)` must equal to the number of the module output. and the final loss is simply sumed.
        metrics: None or list of (output index, 'name', callable)
        callbacks: list of `Callback`
        """
        if callable(optimizer_fn):
            opt = optimizer_fn(self.module.parameters())    # other parameters could be passed by `partial`
        else:
            assert isinstance(optimizer_fn, T.optim.Optimizer)
            opt = optimizer_fn
        # todo: loss_fn is [callable] or [Loss instance]
        if not isinstance(loss_fn, Iterable):
            loss_fn = [loss_fn]
        if callbacks:
            callbacks = CallbackList(callbacks)
            callbacks.set_model(self)
            callbacks.set_optimizer(opt)

        train_ld = self._make_dataloader(training_set, batch_size)
        val_ld = self._make_dataloader(validation_set, batch_size)
        n_target = len(loss_fn)
        training_logging = []
        validation_logging = []
        self.module.to(device)
        for e in range(epochs):
            if self.stop_training:
                break
            # train
            self.module.train()
            running_mean = defaultdict(float) if metrics and train_scoring else None
            running_loss = .0
            pbar = tqdm(enumerate(train_ld), total=len(train_ld), file=sys.stdout)
            for i, batch in pbar:
                opt.zero_grad()
                input, target = self._splitor(batch, n_target, device)
                res = self.module(*input)
                if n_target ==1:
                    res = (res, )
                loss = 0.
                for j in range(n_target):
                    loss += loss_fn[j](res[j], target[j])
                loss.backward()
                running_loss = (running_loss * i + float(loss)) / (1 + i)
                opt.step()
                if running_mean is not None:
                    metrics_output = []
                    for j in range(len(metrics)):
                        k = metrics[j][0]
                        mn = metrics[j][1]
                        running_mean[mn] = (running_mean[mn] * i + metrics[j][2](res[k].detach(), target[k])) / (1 + i)
                        metrics_output.append(f'{mn}={running_mean[mn]:.5f}')
                    metrics_output = ', ' + ', '.join(metrics_output)
                else:
                    metrics_output = ''
                description = f'Epoch [{e}/{epochs}]: loss={running_loss:.5f}' + metrics_output
                pbar.set_description(description)
            training_logging.append({**{'epoch': e, 'loss': running_loss}, **running_mean})
            if validation_set is None:
                continue
            # validation
            self.module.eval()
            running_mean = defaultdict(float) if metrics else None
            running_loss = .0
            pbar = tqdm(enumerate(val_ld), total=len(val_ld), file=sys.stdout)
            self.module.eval()
            for i, batch in pbar:
                input, target = self._splitor(batch, n_target, device)
                res = self.module(*input)
                if n_target == 1:
                    res = (res, )
                loss = 0.
                for j in range(n_target):
                    loss += loss_fn[j](res[j], target[j])
                running_loss = (running_loss * i + float(loss)) / (1 + i)
                if running_mean is not None:
                    metrics_output = []
                    for j in range(len(metrics)):
                        k = metrics[j][0]
                        mn = f'val_{metrics[j][1]}'
                        running_mean[mn] = (running_mean[mn] * i + metrics[j][2](res[k].detach(), target[k])) / (1 + i)
                        metrics_output.append(f'{mn}={running_mean[mn]:.5f}')
                    metrics_output = ', ' + ', '.join(metrics_output)
                else:
                    metrics_output = ''
                description = f'Epoch [{e}/{epochs}]: val_loss={running_loss:.5f}' + metrics_output
                pbar.set_description(description)
            validation_logging.append({**{'epoch': e, 'val_loss': running_loss}, **running_mean})
            if callbacks:
                callbacks.on_epoch_end(training_logging, validation_logging)
        return pd.DataFrame.from_records(training_logging), pd.DataFrame.from_records(validation_logging)
    
    def predict(self, X, batch_size, device='cpu'):
        dl = self._make_dataloader(X, batch_size)
        output = []
        self.module.eval()
        with T.no_grad():
            for i, batch in enumerate(dl):
                input = [c.to(device) for c in batch]
                res = self.module(*input)
                if not isinstance(res, tuple):
                    res = (res, )
                res = (c.cpu().numpy() for c in res)
                output.append(res)
        tmp = tuple(map(np.concatenate, zip(*output)))
        if len(tmp)==1:
            return tmp[0]
        return tmp

    def save(self, fname):
        T.save(self.module.state_dict(), fname)

    def load(self, fname, device):
        self.module.load_state_dict(T.load(fname, map_location=device))
        self.module.to(device)
        return self
