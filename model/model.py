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
        self.model.classifier = nn.Sequential(MultiClassFc(1792))

    def forward(self, x):
        mask, gender, age = self.model(x)
        return mask, gender, age


class Vit_GH(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.Vit = timm.create_model(model_name, pretrained=pretrained)
        self.is_train(self.Vit, False)
        self.Vit.head = MultiClassFc(768)
        self.layer_list = [
            self.Vit.head,
            *[self.Vit.blocks[::-1]],
            self.Vit.patch_embed,
        ]

    def forward(self, x):
        mask_out, gender_out, age_out = self.Vit(x)
        return mask_out, gender_out, age_out

    def is_train(self, module, _train=True):
        if isinstance(module, list):
            for mod in module:
                for m in mod.parameters():
                    m.requires_grad_(_train)
        else:
            for m in module.parameters():
                m.requires_grad_(_train)

    def train_layer(self, epoch):
        self.is_train(self.layer_list[(epoch - 1) % len(self.layer_list)], True)
        self.is_train(self.layer_list[(epoch - 2) % len(self.layer_list)], False)


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
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_class),
        )
