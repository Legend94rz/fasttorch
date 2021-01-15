from torch import nn
from torch.nn import functional as F


class LabelSmoothLoss(nn.Module):
    """
    input: logits. (N, *, C)
    target: one-hot. (N, *, C)
    """
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.argmax(-1, keepdim=True), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
