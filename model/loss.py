from logging import logProcesses
import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ClibGh_loss(outputs, target):
    Loss = nn.CrossEntropyLoss()
    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = torch.split(target, 1, dim=1)

    return (
        0.375 * Loss(mask_out.squeeze(), mask_target.squeeze().to(torch.long))
        + 0.25 * Loss(gender_out.squeeze(), gender_target.squeeze().to(torch.long))
        + 0.375 * Loss(age_out.squeeze(), age_target.squeeze().to(torch.long))
    )
