import torch.nn as nn
import timm
from torchvision import models


class TimmModel(nn.Module):
    def __init__(
        self, model_name="efficientnetv2_rw_s", pretrained=True, num_classes=18
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class TorchVisionModel(nn.Module):
    def __init__(self, model_name="efficientnet_v2_s", pretrained=True, num_classes=18):
        super().__init__()

        self.model = getattr(models, model_name)(pretrained=pretrained)
        n_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
