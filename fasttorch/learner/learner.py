from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Iterable
import numpy as np
import pandas as pd
import sys
import torch as T
from pathlib import Path
from enum import Flag
from ..callbacks import CallbackList
from ..data.tensor_dataloader import TensorDataLoader
from ..misc.seed import seed_everything


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


class LambdaLayer(T.nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Learner:
    __LOCAL_RANK = None

    class Stage(Flag):
        TRAIN = 1
        VALIDATION = 2
        INFERENCE = 3

    @staticmethod
    def init_distributed_training(dummy=False, seed=None):
        if not dummy:
            if Learner.__LOCAL_RANK is None:
                import os
                from torch import distributed as dist
                local_rank = int(os.environ['LOCAL_RANK'])
                T.cuda.set_device(local_rank)
                seed_everything(seed)
                dist.init_process_group(backend='nccl', init_method='env://')
                Learner.__LOCAL_RANK = local_rank
            return Learner.__LOCAL_RANK
        else:
            return 0

    def __init__(self, module):
        if Learner.__LOCAL_RANK is None or isinstance(module, DistributedDataParallel):
            self.module = module
        else:
            self.module = DistributedDataParallel(module.to('cuda'), device_ids=[Learner.__LOCAL_RANK])
        self.stop_training = False
        self.train_ld = self.val_ld = None
        self.nloss = self.opt = self.callbacks = None
        self.validation_logging = []
        self.training_logging = []

    def _walk_through_data(self, stage, cur_epoch, total_epochs, loss_fn, metrics, device, verbose):
        assert stage in (Learner.Stage.TRAIN, Learner.Stage.VALIDATION)
        prev_grad_enabled = T.is_grad_enabled()
        if stage == Learner.Stage.TRAIN:
            self.module.train()
            log_prefix = ''
            dataloader = self.train_ld
            T.set_grad_enabled(True)
        else:
            self.module.eval()
            log_prefix = 'val_'
            dataloader = self.val_ld
            T.set_grad_enabled(False)

        running_mean = defaultdict(float)
        running_loss = .0
        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(cur_epoch)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout, disable=not verbose)
        for i, batch in pbar:
            batch = [c.to(device, non_blocking=True) for c in batch]
            self.callbacks.on_batch_begin(i, batch, self.training_logging, self.validation_logging)
            if stage == Learner.Stage.TRAIN:
                self.opt.zero_grad()
            res = self.compute_forward(batch)
            if not isinstance(res, tuple):
                res = (res, )
            loss = self.compute_losses(loss_fn, res, batch)
            if stage == Learner.Stage.TRAIN:
                loss.backward()
                self.opt.step()
            if metrics:
                detached = tuple(x.detach() for x in res)
                metrics_output = []
                for j in range(len(metrics)):
                    mn = log_prefix + metrics[j][1]
                    running_mean[mn] = (running_mean[mn] * i + self.compute_metric(*metrics[j], detached, batch)) / (1 + i)
                    metrics_output.append(f'{mn}={running_mean[mn]:.5f}')
                metrics_output = ', ' + ', '.join(metrics_output)
            else:
                metrics_output = ''
            self.callbacks.on_batch_end(i, batch, self.training_logging, self.validation_logging)
            running_loss = (running_loss * i + float(loss)) / (1 + i)
            description = f'Epoch [{cur_epoch}/{total_epochs}]: {log_prefix}loss={running_loss:.5f}' + metrics_output
            pbar.set_description(description)
        T.set_grad_enabled(prev_grad_enabled)
        return running_loss, running_mean

    def compute_forward(self, batch_data, stage=Stage.TRAIN):
        if stage == Learner.Stage.TRAIN:
            return self.module(*batch_data[:-self.nloss])
        return self.module(*batch_data)

    def compute_losses(self, loss_fns, forward_results, batch_data):
        """
        :param loss_fns: equals to `loss_fn` in `fit` params
        :param forward_results: tuple. the output of model forward. single output will be wrap to a tuple with len==1.
               if the model only has one objective function while requires multi forward outputs as input,
               use `forward_result[j]` to get j-th forward output component.
        :param batch_data: the output of data_loader's one iter step.
        :return: loss
        """
        target = batch_data[-self.nloss:]
        loss = sum(loss_fns[j](forward_results[j], target[j]) for j in range(self.nloss))
        return loss

    def compute_metric(self, idx, name, func, detached_results, batch_data):
        target = batch_data[-self.nloss:]
        return func(detached_results[idx], target[idx])

    def compute_output(self, detached_results, batch_data):
        return tuple(c.cpu().numpy() for c in detached_results)

    def fit(self, training_set, epochs, batch_size, optimizer_fn, loss_fn=None, metrics=None, validation_set=None, callbacks=None, device='cpu', verbose=True):
        """
        :param training_set: (x1, x2, ..., y1, y2, ...), torch Dataset or DataLoader
        :param epochs: int.
        :param batch_size: int. ignored when training_set is `DataLoader` instance.
        :param optimizer_fn: callable or optim instance.
        :param loss_fn: callable (including loss instance), list of these two type for multi target, or None.
                if loss_fn is a list, the last `len(loss_fn)` components of training_set will be considered as labels respectively.
                besides, `len(loss_fn)` must equal to the number of the module output. and the final loss is simply sumed.
                If `loss_fn` is None, you must override `compute_losses` function.
        :param metrics: None or list of (output index, 'name', callable)
        :param validation_set: the type is same as `training_set`. used for validation.
        :param callbacks: list of `Callback`
        :param device: string, int, or torch.device.
        :param verbose: bool.
        :return: tuple. DataFrame of training and validation log.
        """
        if callable(optimizer_fn):
            self.opt = optimizer_fn(self.module.parameters())    # other parameters could be passed by `partial`
        else:
            assert isinstance(optimizer_fn, T.optim.Optimizer)
            self.opt = optimizer_fn
        # todo: loss_fn is [callable] or [Loss instance]
        if not isinstance(loss_fn, Iterable):
            loss_fn = [loss_fn]
        assert all(callable(x) for x in loss_fn)
        callbacks = [] if callbacks is None else callbacks
        self.callbacks = CallbackList(callbacks)
        self.callbacks.set_model(self)
        self.callbacks.set_params({'optimizer': self.opt})

        self.train_ld = _make_dataloader(training_set, batch_size)
        self.val_ld = _make_dataloader(validation_set, batch_size)
        self.nloss = len(loss_fn)
        self.module.to(device)
        self.stop_training = False
        self.training_logging = []
        self.validation_logging = []
        self.callbacks.on_train_begin()
        for e in range(epochs):
            if self.stop_training:
                break
            self.callbacks.on_epoch_begin(self.training_logging, self.validation_logging)
            running_loss, running_mean = self._walk_through_data(Learner.Stage.TRAIN, e, epochs, loss_fn, metrics, device, verbose)
            self.training_logging.append({**{'epoch': e, 'loss': running_loss}, **running_mean})
            if validation_set is not None:
                running_loss, running_mean = self._walk_through_data(Learner.Stage.VALIDATION, e, epochs, loss_fn, metrics, device, verbose)
                self.validation_logging.append({**{'epoch': e, 'val_loss': running_loss}, **running_mean})
            self.callbacks.on_epoch_end(self.training_logging, self.validation_logging)
        self.callbacks.on_train_end(self.training_logging, self.validation_logging)
        return pd.DataFrame.from_records(self.training_logging), pd.DataFrame.from_records(self.validation_logging)
    
    def predict(self, X, batch_size, device='cpu', verbose=True):
        dl = _make_dataloader(X, batch_size)
        output = []
        self.module.eval()
        with T.no_grad():
            # todo: distributed prediction?
            tmp = None
            if isinstance(self.module, DistributedDataParallel):
                tmp = self.module
                self.module = self.module.module
            self.module.to(device)
            pbar = tqdm(enumerate(dl), total=len(dl), disable=not verbose, file=sys.stdout)
            for i, batch in pbar:
                batch = [c.to(device, non_blocking=True) for c in batch]
                res = self.compute_forward(batch, Learner.Stage.INFERENCE)
                if not isinstance(res, tuple):
                    res = (res, )
                res = self.compute_output(res, batch)
                output.append(res)
            if tmp is not None:
                self.module = tmp
        tmp = tuple(map(np.concatenate, zip(*output)))
        if len(tmp) == 1:
            return tmp[0]
        return tmp

    def save(self, fname):
        m = self.module
        if isinstance(m, DistributedDataParallel):
            m = m.module
        p = Path(fname)
        p.parent.mkdir(parents=True, exist_ok=True)
        T.save(m.state_dict(), fname)

    def load(self, fname, device):
        m = self.module
        if isinstance(m, DistributedDataParallel):
            m = m.module
        m.load_state_dict(T.load(fname, map_location=device))
        m.to(device)
        return self
