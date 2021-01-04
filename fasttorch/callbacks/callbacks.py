from abc import ABC, abstractmethod
from typing import Iterable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial


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
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
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
