import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ClibGh_loss(outputs, target):
    criterion_mask = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = torch.split(target, 1, dim=1)

    return (
        0.375 * criterion_mask(mask_out.squeeze(), mask_target.squeeze().to(torch.long))
        + 0.25
        * criterion_gender(gender_out.squeeze(), gender_target.squeeze().to(torch.long))
        + 0.375 * criterion_age(age_out.squeeze(), age_target.squeeze().to(torch.long))
    )
