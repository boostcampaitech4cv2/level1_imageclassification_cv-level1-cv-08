import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def multilabel_loss(outputs, target):
    Loss_mask = nn.CrossEntropyLoss(weight=torch.tensor(([1.4, 7.0, 7.0]))).cuda()
    Loss_gender = nn.CrossEntropyLoss(weight=torch.tensor(([1.6285, 2.5912]))).cuda()
    Loss_age = nn.CrossEntropyLoss(
        weight=torch.tensor(([2.1077, 2.2005, 14.0625]))
    ).cuda()

    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = target

    return (
        0.375 * Loss_mask(mask_out.squeeze(), mask_target.squeeze().to(torch.long))
        + 0.25
        * Loss_gender(gender_out.squeeze(), gender_target.squeeze().to(torch.long))
        + 0.375 * Loss_age(age_out.squeeze(), age_target.squeeze().to(torch.long))
    )


def metrics_loss(loss_name):
    return losses(getattr, loss_name)
