import torch.nn as nn
import timm
from torchvision import models


class TimmModel(nn.Module):
    def __init__(
        self, model_name="efficientnetv2_rw_s", pretrained=True, num_classes=18
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

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


class TimmModelMulti(nn.Module):
    def __init__(self, model_name="efficientnetv2_rw_s", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        self.model.classifier = nn.Sequential(
            MultiClassFc(self.model.classifier.in_features)
        )

    def forward(self, x):
        mask, gender, age = self.model(x)
        return mask, gender, age


class MultiClassFc(nn.Module):
    def __init__(self, backbone_dim):
        super().__init__()
        self.mask_out = self.make_out_layer(backbone_dim, 3)
        self.gender_out = self.make_out_layer(backbone_dim, 2)
        self.age_out = self.make_out_layer(backbone_dim, 6)

    def forward(self, x):
        mask_out = self.mask_out(x)
        gender_out = self.gender_out(x)
        age_out = self.age_out(x)
        return mask_out, gender_out, age_out

    def make_out_layer(self, backbone_dim, num_class):
        return nn.Sequential(
            nn.Linear(backbone_dim, num_class),
        )
