import torch
import torch.nn.functional as F
import torch.nn as nn
from base import BaseModel

def nll_loss(output, target):
    return F.nll_loss(output, target)

def ce_loss(output, target):
    return F.cross_entropy(output, target)

def bce_with_logits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

# class FocalLoss(BaseModel):
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2., reduction='mean'):
#         # super().__init__()        
#         nn.Module.__init__(self)
#         self.weight = weight
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input_tensor, target_tensor):
#         log_prob = F.log_softmax(input_tensor, dim=-1)
#         prob = torch.exp(log_prob)
#         return F.nll_loss(
#             ((1 - prob) ** self.gamma) * log_prob,
#             target_tensor,
#             weight=self.weight,
#             reduction=self.reduction
#         )

def fc_loss(output, target):
    # return FocalLoss(output, target)
    gamma = 2
    weight = None
    reduction = 'mean'
    log_prob = F.log_softmax(output, dim=-1)
    prob = torch.exp(log_prob)
    return F.nll_loss(
        ((1 - prob) ** gamma) * log_prob,
        target, weight, reduction
    )

# def fc_loss(output, target, alpha=1, gamma=2):
#     BCE_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
#     pt = torch.exp(-BCE_loss) # prevents nans when probability 0
#     F_loss = alpha * (1-pt)**gamma * BCE_loss
#     return focal_loss.mean()