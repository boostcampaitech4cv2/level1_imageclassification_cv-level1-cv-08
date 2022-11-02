import torch.nn.functional as F
import torch.nn as nn
import torch


def multilabel_loss(outputs, target):
    Loss_mask = nn.CrossEntropyLoss(weight=torch.tensor(([1.4, 7.0, 7.0]))).cuda()
    Loss_gender = nn.CrossEntropyLoss(weight=torch.tensor(([1.6285, 2.5912]))).cuda()
    Loss_age = nn.CrossEntropyLoss().cuda()

    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = target

    return (
        0.375 * Loss_mask(mask_out.squeeze(), mask_target.squeeze().to(torch.long))
        + 0.25
        * Loss_gender(gender_out.squeeze(), gender_target.squeeze().to(torch.long))
        + 0.375 * Loss_age(age_out.squeeze(), age_target.squeeze().to(torch.long))
    )


def losses(loss_name, output, target):
    if loss_name == "nll_loss":
        return F.nll_loss(output, target)
    elif loss_name == "ce_loss":
        return F.cross_entropy(output, target)
    elif loss_name == "multilabel_loss":
        return multilabel_loss(output, target)
    else:
        return getattr(losses, loss_name)()(output, target)
