import torch.nn.functional as F
import torch.nn as nn
import torch
from pytorch_metric_learning import losses


def multilabel_loss(outputs, target):
    loss_mask = nn.CrossEntropyLoss(weight=torch.tensor(([1.4, 7.0, 7.0]))).cuda()
    loss_gender = nn.CrossEntropyLoss(weight=torch.tensor(([1.6285, 2.5912]))).cuda()
    loss_age = nn.CrossEntropyLoss(
        weight=torch.tensor(([2.8723, 3.9589, 4.5075, 14.0625, 14.0625, 28.4211]))
    ).cuda()

    mask_out, gender_out, age_out = outputs
    mask_target, gender_target, age_target = target

    return (
        0.375 * loss_mask(mask_out, mask_target)
        + 0.25 * loss_gender(gender_out, gender_target)
        + 0.375 * loss_age(age_out, age_target)
    )


def all_loss(loss_name, output, target, inner_weight, loss_weight):
    if inner_weight:
        inner_weight = torch.tensor((inner_weight)).cuda()
    if loss_name == "ce_loss":
        return F.cross_entropy(output, target, weight=inner_weight) * loss_weight
    elif loss_name == "focal":
        return create_criterion("focal")(output, target) * loss_weight
    elif loss_name == "label_smoothing":
        return create_criterion("label_smoothing")(output, target) * loss_weight
    elif loss_name == "f1":
        return create_criterion("f1")(output, target) * loss_weight
    else:
        return getattr(all_loss, loss_name)()(output, target) * loss_weight


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "f1": F1Loss,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion
