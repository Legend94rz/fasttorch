from abc import ABC, abstractmethod
from typing import Iterable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
import warnings
import numpy as np
from pathlib import Path


class BaseCallback(ABC):
    def set_model(self, learner):
        self.learner = learner

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def on_epoch_end(self, training_log, validation_log):
        pass


class ReduceLROnPlateauCallback(BaseCallback):
    def __init__(self, monitor='val_loss', mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self.monitor = monitor
        self.__schdule_fn = partial(ReduceLROnPlateau, mode=mode, factor=factor, patience=patience,
                 verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
                 cooldown=cooldown, min_lr=min_lr, eps=eps)

    def set_optimizer(self, optimizer):
        super().set_optimizer(optimizer)
        self.schedule = self.__schdule_fn(optimizer)

    def on_epoch_end(self, training_log, validation_log):
        cur_log = {**training_log[-1], **validation_log[-1]}
        self.schedule.step(cur_log[self.monitor])


class CallbackList(BaseCallback):
    def __init__(self, callbacks):
        assert isinstance(callbacks, Iterable)
        self.callbacks = callbacks

    def set_model(self, learner):
        for cbk in self.callbacks:
            cbk.set_model(learner)

    def set_optimizer(self, optimizer):
        for cbk in self.callbacks:
            cbk.set_optimizer(optimizer)

    def on_epoch_end(self, training_log, validation_log):
        for cbk in self.callbacks:
            cbk.on_epoch_end(training_log, validation_log)


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, monitor='val_loss', patience=1, mode='min', verbose=False):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.epoch = -1
        self.verbose = verbose
        if mode == 'min':
            self.opt = float('inf')
        else:
            self.opt = float('-inf')

    def on_epoch_end(self, training_log, validation_log):
        cur_log = {**training_log[-1], **validation_log[-1]}
        if (self.mode == 'min' and cur_log[self.monitor] < self.opt) or (self.mode == 'max' and cur_log[self.monitor] > self.opt):
            self.opt = cur_log[self.monitor]
            self.epoch = len(training_log)
        else:
            if len(training_log) - self.epoch > self.patience:
                self.learner.stop_training = True
                if self.verbose:
                    print("Early stopping")


class ModelCheckpoint(BaseCallback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, bool.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {min, max}.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=False, save_best_only=False, mode='min', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.last_save_file = None
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, training_log, validation_log):
        logs = {**training_log[-1], **validation_log[-1]}
        epoch = len(training_log)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(**logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose:
                            print(f'Epoch {epoch:05d}: {self.monitor} improved from {self.best:0.5f} to {current:0.5f}, saving model to {filepath}')
                        self.best = current
                        if self.last_save_file is not None and Path(self.last_save_file).exists():
                            Path(self.last_save_file).unlink()
                        self.learner.save(filepath)
                        self.last_save_file = filepath
                    else:
                        if self.verbose:
                            print(f'Epoch {epoch:05d}: {self.monitor} did not improve')
            else:
                if self.verbose:
                    print(f'Epoch {epoch:05d}: saving model to {filepath}')
                self.learner.save(filepath)
                self.last_save_file = filepath
