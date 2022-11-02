import torch.nn.functional as F
from pytorch_metric_learning import losses

def nll_loss(output, target):
    return F.nll_loss(output, target)


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def losses(loss_name, output, target):
    if loss_name == "nll_loss":
        return F.nll_loss(output, target)
    elif loss_name == "ce_loss":
        return F.cross_entropy(output, target)
    else:
        return getattr(losses, loss_name)()(output, target)


    