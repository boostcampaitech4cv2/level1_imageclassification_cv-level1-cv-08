import torch
from sklearn.metrics import f1_score


def accuracy(pred, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = sum(torch.sum(pred[:, i] == target).item() for i in range(k))
    return correct / len(target)


def f1(pred, target):
    with torch.no_grad():
        return f1_score(target.cpu(), pred.cpu(), average="macro")

def mask_accuracy(pred, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred//6 == target//6).item()
    return correct / len(target)

def gender_accuracy(pred, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred%6//3 == target%6//3).item()
    return correct / len(target)

def age_accuracy(pred, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred%6%3 == target%6%3).item()
    return correct / len(target)