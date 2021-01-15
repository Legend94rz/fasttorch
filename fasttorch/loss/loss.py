from torch import nn
from torch.nn import functional as F
import torch as T


class LabelSmoothLoss(nn.Module):
    """
    input: (N, *, C).
    target: if `target_type` == 'one-hot': one-hot. (N, *, C);
            if `target_type` == 'sparse' : (N, *), 0 ~ C-1
    """
    def __init__(self, smoothing=0.0, input_type='logit', target_type='sparse'):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        assert input_type in ['logit', 'prob']
        assert target_type in ['sparse', 'one-hot']
        if input_type == 'logit':
            self.log_prob_fn = lambda x: F.log_softmax(x, dim=-1)
        else:
            self.log_prob_fn = lambda x: T.log(x)
        if target_type == 'sparse':
            self.weight_fn = lambda i, t, m: (i.new_ones(i.size()) * m / (i.size(-1) - 1.)).scatter_(-1, t.unsqueeze(-1), 1. - m)
        else:
            self.weight_fn = lambda i, t, m: (i.new_ones(i.size()) * m / (i.size(-1) - 1.)).scatter_(-1, t.argmax(-1, keepdim=True), 1. - m)

    def forward(self, input, target):
        log_prob = self.log_prob_fn(input)
        weight = self.weight_fn(input, target, self.smoothing)
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class BinaryLabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0, input_type='logit'):
        super().__init__()
        self.smoothing = smoothing
        assert input_type in ['logit', 'prob']
        if input_type == 'logit':
            self.fn = lambda x, y: F.binary_cross_entropy_with_logits(x, y)
        else:
            self.fn = lambda x, y: F.binary_cross_entropy(x, y)
        self.weight_fn = lambda t, m: (1.-2*m)*t

    def forward(self, input, target):
        w = self.weight_fn(target, self.smoothing)
        return self.fn(input, w)
