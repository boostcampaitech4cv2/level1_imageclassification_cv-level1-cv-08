import torch


def accuracy(output, target):
    mask_target, gender_target, age_target = torch.split(target, 1, dim=1)
    mask_out, gender_out, age_out = output
    output = (
        mask_out.data.max(1, keepdim=True)[1] * 6
        + gender_out.data.max(1, keepdim=True)[1] * 2
        + age_out.data.max(1, keepdim=True)[1]
    )
    target = (
        mask_target.data.max(1, keepdim=True)[1] * 6
        + gender_target.data.max(1, keepdim=True)[1] * 2
        + age_target.data.max(1, keepdim=True)[1]
    )
    with torch.no_grad():
        assert output.shape[0] == len(target)
        correct = 0
        correct += torch.sum(output == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
