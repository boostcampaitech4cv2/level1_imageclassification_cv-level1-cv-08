from this import s
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import timm


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Res18(BaseModel):
    def __init__(self):
        super().__init__()
        self.res18 = timm.create_model("resnet18d", pretrained=True, num_classes=18)

    def forward(self, x):
        x = self.res18(x)
        return F.log_softmax(x, dim=1)


class ClibGh(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        import clip

        model, _ = clip.load("RN50", device="cuda")

        self.device = device
        self.dtype = model.visual.conv1.weight.dtype

        # img
        self.encode_image = model.visual

        # text
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

        # freeze
        self.is_train(self.encode_image, _train=False)
        self.is_train(self.token_embedding, _train=False)
        self.is_train(self.transformer, _train=False)
        self.is_train(self.ln_final, _train=False)
        self.positional_embedding.requires_grad_(False)
        self.text_projection.requires_grad_(False)
        # train

        self.mask_text = [
            "This person is wearing a mask correctly.",
            "This person is wearing a mask incorrectly.",
            "This person is not wearing a mask.",
        ]
        self.gender_text = ["This photo is a man.", "This photo is a woman."]
        self.age_text = [
            "This person is under 30 years old.",
            "This person is over 30 and under 60",
            "This person is over 60 years old.",
        ]

        self.mask_token = clip.tokenize(self.mask_text).to(self.device)
        self.gender_token = clip.tokenize(self.gender_text).to(self.device)
        self.age_token = clip.tokenize(self.age_text).to(self.device)

    def forward(self, x):
        img_feature = self.encode_image(x).unsqueeze(1)
        img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)

        mask_feature = self.encode_text(self.mask_token).to(self.device)
        mask_feature = mask_feature / mask_feature.norm(dim=1, keepdim=True)

        gender_feature = self.encode_text(self.gender_token).to(self.device)
        gender_feature = gender_feature / gender_feature.norm(dim=1, keepdim=True)

        age_feature = self.encode_text(self.age_token).to(self.device)
        age_feature = age_feature / age_feature.norm(dim=1, keepdim=True)

        return img_feature, mask_out, gender_out, age_out

    def is_train(self, module, _train=True):
        for m in module.parameters():
            m.requires_grad_(_train)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x.shape = [batch_size, n_ctx, transformer.width]
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
