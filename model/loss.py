import torch.nn.functional as F
from pytorch_metric_learning import losses


def nll_loss(output, target):
    return F.nll_loss(output, target)


def multilabel_loss(outputs, target):
    loss_mask = nn.CrossEntropyLoss(weight=torch.tensor(([1.4, 7.0, 7.0]))).cuda()
    loss_gender = nn.CrossEntropyLoss(weight=torch.tensor(([1.6285, 2.5912]))).cuda()
    loss_age = nn.CrossEntropyLoss(
        weight=torch.tensor(([2.8723, 3.9589, 4.5075, 14.0625, 14.0625, 28.4211]))
    ).cuda()


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def all_loss(loss_name, output, target):
    if loss_name == "nll_loss":
        return F.nll_loss(output, target)
    elif loss_name == "ce_loss":
        return F.cross_entropy(output, target)
    else:
        return getattr(losses, loss_name)()(output, target)
