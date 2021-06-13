from sklearn.metrics import roc_auc_score, f1_score as sk_f1_score
import torch as T


def binary_accuracy(input: T.Tensor, target: T.Tensor):
    return (input.float().round() == target).float().mean().cpu().numpy()


def binary_accuracy_with_logits(input, target):
    return ((input>0).float() == target).float().mean().cpu().numpy()


def sparse_categorical_accuracy(input, target):
    """
    :param input: (N, c) Tensor
    :param target: (N, ) Tensor
    :return: numpy float
    """
    return (input.argmax(dim=-1) == target).float().mean().cpu().numpy()


def auc_score(input, target, **kwargs):
    return roc_auc_score(target.cpu().numpy(), input.cpu().numpy(), **kwargs)


def mean_squared_error(input: T.Tensor, target: T.Tensor):
    return ((input - target)**2).mean()


def mean_absolute_error(input, target):
    return (input - target).abs().mean()


def f1_score(input: T.Tensor, target: T.Tensor, **kwargs):
    return sk_f1_score(target.cpu().numpy().astype('int'), input.cpu().numpy().astype('int'), **kwargs)
