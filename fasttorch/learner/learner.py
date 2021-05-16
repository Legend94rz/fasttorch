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
import inspect as isp
from ..callbacks import CallbackList
from ..data.tensor_dataloader import TensorDataLoader
from ..misc.seed import seed_everything


class LambdaLayer(T.nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class TimeDistributed(T.nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # B T ...
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, *x.shape[2:])
        y = self.module(x_reshape)  # B*T ...
        return y.view(-1, x.shape[1], *y.shape[1:])


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

    @staticmethod
    def _make_dataloader(dataset, batch_size, **kwargs):
        if isinstance(dataset, DataLoader) or isinstance(dataset, TensorDataLoader):
            return dataset

        dl_kwargs = {k: v for k, v in kwargs.items() if k not in ('self', 'kwargs', 'batch_size') and k in isp.signature(DataLoader.__init__).parameters}
        if isinstance(dataset, Dataset):
            pass
        elif isinstance(dataset, np.ndarray) or isinstance(dataset, T.Tensor):
            dataset = TensorDataset(T.tensor(dataset))
        elif isinstance(dataset, Iterable):
            dataset = TensorDataset(*[T.tensor(a) for a in dataset])
        else:
            raise NotImplementedError
        if 'sampler' not in dl_kwargs and Learner.__LOCAL_RANK is not None and '__INFERENCE__' not in kwargs:
            dl_kwargs['sampler'] = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, **dl_kwargs)

    @staticmethod
    def _move_batch_to_device(batch, device):
        # list or tuple of `Tensor`;
        # single `Tensor` or `np.ndarray`.
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [c.to(device, non_blocking=True) for c in batch]
        elif isinstance(batch, T.Tensor):
            batch = [batch.to(device, non_blocking=True)]
        elif isinstance(batch, np.ndarray):
            batch = [T.tensor(batch, device=device)]
        else:
            raise NotImplementedError
        return batch

    def __init__(self, module, optimizer_fn=None, loss_fn=None):
        """
        :param module:
        :param optimizer_fn: callable or optim instance.
        :param loss_fn: callable (including loss instance), list of callable for multi-target module, or None.
                if loss_fn is a list, the last `len(loss_fn)` components of `training_set` will be considered as labels respectively.
                besides, `len(loss_fn)` must equal to the number of the module output. and the final loss is simply sumed.
                If `loss_fn` is None, you must override `compute_losses` function for training.
        """
        if Learner.__LOCAL_RANK is None or isinstance(module, DistributedDataParallel):
            self.module = module
        else:
            module_ = T.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            self.module = DistributedDataParallel(module_.to('cuda'), device_ids=[Learner.__LOCAL_RANK])
        self.stop_training = False
        self.train_ld = self.val_ld = None
        self.nloss = self.opt = None
        self.validation_logging = []
        self.training_logging = []
        if optimizer_fn:
            if callable(optimizer_fn):
                self.opt = optimizer_fn(self.module.parameters())    # other parameters could be passed by `partial`
            else:
                assert isinstance(optimizer_fn, T.optim.Optimizer)
                self.opt = optimizer_fn
        else:
            self.opt = None
        if loss_fn:
            if not isinstance(loss_fn, Iterable):
                self.loss_fn = [loss_fn]
            assert all(callable(x) or None for x in self.loss_fn)
            self.nloss = len(self.loss_fn)
        else:
            self.nloss = None
            self.loss_fn = None

    def _iter_one_batch(self, stage, batch, metrics):
        if stage == Learner.Stage.TRAIN:
            self.opt.zero_grad()
        res = self.compute_forward(batch, stage)
        if not isinstance(res, tuple):  # if single output
            res = (res,)
        if stage != Learner.Stage.INFERENCE:
            loss = self.compute_losses(res, batch)
            if stage == Learner.Stage.TRAIN:
                loss.backward()
                self.opt.step()
            metrics_result = {}
            if metrics:
                detached = tuple(x.detach() for x in res)
                for j in range(len(metrics)):
                    mn = metrics[j][1]
                    metrics_result[mn] = self.compute_metric(*metrics[j], detached, batch)
            return loss, metrics_result
        else:
            return self.compute_output(res, batch)

    def _walk_through_data(self, stage, cur_epoch, total_epochs, metrics, callbacks, device, verbose):
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
            batch = Learner._move_batch_to_device(batch, device)
            metrics_output = []
            callbacks.on_batch_begin(i, batch, self.training_logging, self.validation_logging)
            loss, metrics_result = self._iter_one_batch(stage, batch, metrics)
            for k, v in metrics_result.items():
                mn = log_prefix + k
                running_mean[mn] = (running_mean[mn] * i + v) / (1 + i)
                metrics_output.append(f'{mn}={running_mean[mn]:.5f}')
            metrics_output = (', ' if metrics else '') + ', '.join(metrics_output)
            callbacks.on_batch_end(i, batch, self.training_logging, self.validation_logging)
            running_loss = (running_loss * i + float(loss)) / (1 + i)
            description = f'Epoch [{cur_epoch}/{total_epochs}]: {log_prefix}loss={running_loss:.5f}' + metrics_output
            pbar.set_description(description)
        T.set_grad_enabled(prev_grad_enabled)
        del batch, loss
        return running_loss, running_mean

    def compute_forward(self, batch_data, stage=Stage.TRAIN):
        if stage == Learner.Stage.INFERENCE:
            return self.module(*batch_data)
        return self.module(*batch_data[:-self.nloss])

    def compute_losses(self, forward_results, batch_data):
        """
        :param forward_results: tuple. the output of model forward. single output will be wrap to a tuple with len==1.
               use `forward_result[j]` to get j-th forward output component.
        :param batch_data: the output of data_loader's one iter step.
        :return: loss
        """
        target = batch_data[-self.nloss:]
        loss = sum(self.loss_fn[j](forward_results[j], target[j]) for j in range(self.nloss))
        return loss

    def compute_metric(self, idx, name, func, detached_results, batch_data):
        target = batch_data[-self.nloss:]
        return func(detached_results[idx], target[idx])

    def compute_output(self, detached_results, batch_data):
        return tuple(c.cpu().numpy() for c in detached_results)

    def fit_one_batch(self, batch, metrics, device='cpu'):
        batch = self._move_batch_to_device(batch, device)
        self.module.train()
        with T.enable_grad():
            loss, metrics_result = self._iter_one_batch(Learner.Stage.TRAIN, batch, metrics)
            return loss, metrics_result

    def valid_one_batch(self, batch, metrics, device='cpu'):
        batch = self._move_batch_to_device(batch, device)
        self.module.eval()
        with T.no_grad():
            loss, metrics_result = self._iter_one_batch(Learner.Stage.VALIDATION, batch, metrics)
            return loss, {f'val_{k}': v for k, v in metrics_result.items()}

    def predict_one_batch(self, batch, metrics, device='cpu'):
        batch = self._move_batch_to_device(batch, device)
        self.module.eval()
        with T.no_grad():
            output = self._iter_one_batch(Learner.Stage.INFERENCE, batch, metrics)
            return output

    def fit(self, training_set, epochs, batch_size, metrics=None, validation_set=None, callbacks=None, device='cpu', verbose=True, **kwargs):
        """
        :param training_set: (x1, x2, ..., y1, y2, ...), torch Dataset or DataLoader
        :param epochs: int.
        :param batch_size: int. ignored when training_set is `DataLoader` instance.
        :param metrics: None or list of (output index, 'name', callable)
        :param validation_set: the type is same as `training_set`. used for validation.
        :param callbacks: list of `Callback`
        :param device: string, int, or torch.device.
        :param verbose: bool.
        :param kwargs: will passed to build dataloader
        :return: tuple. DataFrame of training and validation log.
        """
        assert self.opt is not None, 'No optimizer.'
        callbacks = [] if callbacks is None else callbacks
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.set_params({'optimizer': self.opt})

        self.train_ld = Learner._make_dataloader(training_set, batch_size, **kwargs)
        self.val_ld = Learner._make_dataloader(validation_set, batch_size, **kwargs)
        self.module.to(device)
        self.stop_training = False
        self.training_logging = []
        self.validation_logging = []
        callbacks.on_train_begin()
        for e in range(epochs):
            if self.stop_training:
                break
            callbacks.on_epoch_begin(self.training_logging, self.validation_logging)
            running_loss, running_mean = self._walk_through_data(Learner.Stage.TRAIN, e, epochs, metrics, callbacks, device, verbose)
            self.training_logging.append({**{'epoch': e, 'loss': running_loss}, **running_mean})
            if validation_set is not None:
                running_loss, running_mean = self._walk_through_data(Learner.Stage.VALIDATION, e, epochs, metrics, callbacks, device, verbose)
                self.validation_logging.append({**{'epoch': e, 'val_loss': running_loss}, **running_mean})
            callbacks.on_epoch_end(self.training_logging, self.validation_logging)
        callbacks.on_train_end(self.training_logging, self.validation_logging)
        return pd.DataFrame.from_records(self.training_logging), pd.DataFrame.from_records(self.validation_logging)
    
    def predict(self, X, batch_size, device='cpu', verbose=True, **kwargs):
        kwargs['__INFERENCE__'] = True
        dl = Learner._make_dataloader(X, batch_size, **kwargs)
        output = []
        self.module.eval()
        with T.no_grad():
            # todo: distributed prediction?
            #mbackup = None
            #if isinstance(self.module, DistributedDataParallel):
            #    mbackup = self.module
            #    self.module = self.module.module
            self.module.to(device)
            pbar = tqdm(enumerate(dl), total=len(dl), disable=not verbose, file=sys.stdout)
            for i, batch in pbar:
                batch = Learner._move_batch_to_device(batch, device)
                res = self._iter_one_batch(Learner.Stage.INFERENCE, batch, None)
                output.append(res)
            #if mbackup is not None:
            #    self.module = mbackup
        del batch, dl
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

    def load(self, fname, map_location=None):
        m = self.module
        if isinstance(m, DistributedDataParallel):
            m = m.module
        m.load_state_dict(T.load(fname, map_location=map_location))
        return self
