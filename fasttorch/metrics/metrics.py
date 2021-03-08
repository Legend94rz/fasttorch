from sklearn.metrics import roc_auc_score
import torch as T


def binary_accuracy(input: T.Tensor, target: T.Tensor):
    return (input.float().round() == target).float().mean().cpu().numpy()


def binary_accuracy_with_logits(input, target):
    return ((input>0).float() == target).float().mean().cpu().numpy()


def categorical_accuracy(input, target):
    return (input.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().cpu().numpy()


def auc_score(input, target, **kwargs):
    return roc_auc_score(target.cpu().numpy(), input.cpu().numpy(), **kwargs)


def mean_squared_error(input: T.Tensor, target: T.Tensor):
    return ((input - target)**2).mean()


def mean_absolute_error(y_true, y_pred):
    return (y_pred - y_true).abs().mean()
