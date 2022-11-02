import torch.nn.functional as F
import torch.nn as nn
import torch


def multilabel_loss(outputs, target):
    loss_mask = nn.CrossEntropyLoss(weight=torch.tensor(([1.4, 7.0, 7.0]))).cuda()
    loss_gender = nn.CrossEntropyLoss(weight=torch.tensor(([1.6285, 2.5912]))).cuda()
    loss_age = nn.CrossEntropyLoss().cuda()

    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = target

    return (
        0.375 * loss_mask(mask_out, mask_target)
        + 0.25 * loss_gender(gender_out, gender_target)
        + 0.375 * loss_age(age_out, age_target)
    )


def all_loss(loss_name, output, target):
    if loss_name == "nll_loss":
        return F.nll_loss(output, target)
    elif loss_name == "ce_loss":
        return F.cross_entropy(output, target)
    elif loss_name == "multilabel_loss":
        return multilabel_loss(output, target)
    else:
        return getattr(all_loss, loss_name)()(output, target)
