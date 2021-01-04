from sklearn.metrics import roc_auc_score


def binary_accuracy(input, target):
    return (input.float().round() == target).float().mean()


def binary_accuracy_with_logits(input, target):
    return ((input>0).float() == target).float().mean()


def categorical_accuracy(input, target):
    return (input.argmax(dim=-1) == target.argmax(dim=-1)).float().mean()


def auc_score(input, target, **kwargs):
    return roc_auc_score(target.cpu().numpy(), input.cpu().numpy(), **kwargs)
