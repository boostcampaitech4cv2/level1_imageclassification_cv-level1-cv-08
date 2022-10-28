import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ClibGh_loss(outputs, target):
    criterion_mask = F.cross_entropy
    criterion_gender = F.cross_entropy
    criterion_age = F.cross_entropy
    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = torch.split(target, 1, dim=1)
    return (
        0.375 * criterion_mask(mask_out, mask_target.squeeze())
        + 0.25 * criterion_gender(gender_out, gender_target.squeeze())
        + 0.375 * criterion_age(age_out, age_target.squeeze())
    ) * 1e-8
