import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def macro_f1(output, target):
    def f1(output, target, pivot=0):
        if pivot not in output or pivot not in target:
            return None

        epsilon = 1e-7
        precision = (
            torch.logical_and(output == pivot, target == pivot).sum().item()
            / torch.sum(output == pivot).item()
        )
        recall = (
            torch.logical_and(output == pivot, target == pivot).sum().item()
            / torch.sum(target == pivot).item()
        )
        return 2 * (precision * recall) / (precision + recall + epsilon)

    mask_target, gender_target, age_target = torch.split(target, 1, dim=1)
    mask_out, gender_out, age_out = output
    output = (
        mask_out.data.max(1, keepdim=True)[1] * 6
        + gender_out.data.max(1, keepdim=True)[1] * 2
        + age_out.data.max(1, keepdim=True)[1]
    ).squeeze()
    target = (mask_target * 6 + gender_target * 2 + age_target).squeeze()

    macrof1 = [
        f1(output, target, i) for i in range(18) if f1(output, target, i) != None
    ]
    return sum(macrof1) / 18
