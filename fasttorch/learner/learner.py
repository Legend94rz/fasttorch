import torch as T
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import sys
from typing import Iterable
import numpy as np
from collections import defaultdict
import pandas as pd
from ..data.tensor_dataloader import TensorDataLoader
from ..callbacks import CallbackList


def _splitor(batch, n_target, device):
    batch = [c.to(device, non_blocking=True) for c in batch]
    return batch[:-n_target], batch[-n_target:]


def _make_dataloader(dataset, batch_size):
    if dataset is None:
        return None
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, batch_size=batch_size, pin_memory=False)
    elif isinstance(dataset, DataLoader) or isinstance(dataset, TensorDataLoader):
        return dataset
    elif isinstance(dataset, np.ndarray) or isinstance(dataset, T.Tensor):
        return DataLoader(TensorDataset(T.tensor(dataset)), batch_size=batch_size, pin_memory=False)
    elif isinstance(dataset, Iterable):
        dataset = [T.tensor(a) for a in dataset]
        return DataLoader(TensorDataset(*dataset), batch_size=batch_size, pin_memory=False)
    else:
        raise NotImplementedError


class Learner:
    def __init__(self, module):
        self.module = module
        self.stop_training = False

    def _walk_through_data(self, split, cur_epoch, total_epochs, opt, loss_fn, metrics, device, verbose):
        assert (split in ('train', 'val'))
        log_prefix = '' if split == 'train' else 'val_'
        dataloader = self.train_ld if split == 'train' else self.val_ld
        if split == 'train':
            self.module.train()
        else:
            self.module.eval()

        running_mean = defaultdict(float) if metrics else None
        running_loss = .0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout, disable=not verbose)
        for i, batch in pbar:
            if split == 'train':
                opt.zero_grad()
            feature, target = _splitor(batch, self.n_target, device)
            res = self.module(*feature)
            if self.n_target == 1:
                res = (res,)
            loss = 0.
            for j in range(self.n_target):
                loss += loss_fn[j](res[j], target[j])
            running_loss = (running_loss * i + float(loss)) / (1 + i)
            if split == 'train':
                loss.backward()
                opt.step()
            if running_mean is not None:
                metrics_output = []
                for j in range(len(metrics)):
                    k = metrics[j][0]
                    mn = log_prefix + metrics[j][1]
                    running_mean[mn] = (running_mean[mn] * i + metrics[j][2](res[k].detach(), target[k])) / (1 + i)
                    metrics_output.append(f'{mn}={running_mean[mn]:.5f}')
                metrics_output = ', ' + ', '.join(metrics_output)
            else:
                metrics_output = ''
            description = f'Epoch [{cur_epoch}/{total_epochs}]: {log_prefix}loss={running_loss:.5f}' + metrics_output
            pbar.set_description(description)
        return running_loss, running_mean

    def fit(self, training_set, epochs, batch_size, optimizer_fn, loss_fn, metrics=None, validation_set=None, callbacks=None, device='cpu', verbose=True):
        """
        training_set: (x1, x2, ..., y1, y2, ...), torch Dataset or DataLoader
        batch_size: ignored when training_set is `DataLoader` instance.T
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
        callbacks = [] if callbacks is None else callbacks
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.set_params({'optimizer': opt})

        self.train_ld = _make_dataloader(training_set, batch_size)
        self.val_ld = _make_dataloader(validation_set, batch_size)
        self.n_target = len(loss_fn)
        training_logging = []
        validation_logging = []
        self.module.to(device)
        self.stop_training = False
        for e in range(epochs):
            if self.stop_training:
                break
            running_loss, running_mean = self._walk_through_data('train', e, epochs, opt, loss_fn, metrics, device, verbose)
            training_logging.append({**{'epoch': e, 'loss': running_loss}, **running_mean})
            if validation_set is not None:
                running_loss, running_mean = self._walk_through_data('val', e, epochs, None, loss_fn, metrics, device, verbose)
                validation_logging.append({**{'epoch': e, 'val_loss': running_loss}, **running_mean})
            callbacks.on_epoch_end(training_logging, validation_logging)
        callbacks.on_train_end(training_logging, validation_logging)
        return pd.DataFrame.from_records(training_logging), pd.DataFrame.from_records(validation_logging)
    
    def predict(self, X, batch_size, device='cpu'):
        dl = _make_dataloader(X, batch_size)
        output = []
        self.module.eval()
        with T.no_grad():
            for i, batch in enumerate(dl):
                input = [c.to(device, non_blocking=True) for c in batch]
                res = self.module(*input)
                if not isinstance(res, tuple):
                    res = (res, )
                res = (c.cpu().numpy() for c in res)
                output.append(res)
        tmp = tuple(map(np.concatenate, zip(*output)))
        if len(tmp) == 1:
            return tmp[0]
        return tmp

    def save(self, fname):
        T.save(self.module.state_dict(), fname)

    def load(self, fname, device):
        self.module.load_state_dict(T.load(fname, map_location=device))
        self.module.to(device)
        return self
