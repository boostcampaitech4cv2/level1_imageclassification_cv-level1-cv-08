import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ClibGh_loss(outputs, target):
    criterion_mask = nn.CrossEntropyLoss(reduction="sum")
    criterion_gender = nn.CrossEntropyLoss(reduction="sum")
    criterion_age = nn.CrossEntropyLoss(reduction="sum")
    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = torch.split(target, 1, dim=1)

    return (
        50 * criterion_mask(mask_out.squeeze(), mask_target.squeeze().to(torch.long))
        + 50
        * criterion_gender(gender_out.squeeze(), gender_target.squeeze().to(torch.long))
        + 50 * criterion_age(age_out.squeeze(), age_target.squeeze().to(torch.long))
    )
