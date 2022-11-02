import torch.nn.functional as F
from pytorch_metric_learning import losses

def nll_loss(output, target):
    return F.nll_loss(output, target)


def ce_loss(output, target):
    return F.cross_entropy(output, target)

def metrics_loss(loss_name):
    return losses(getattr, loss_name)