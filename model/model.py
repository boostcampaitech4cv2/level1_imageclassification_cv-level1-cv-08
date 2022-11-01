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


class Res(BaseModel):
    def __init__(self):
        super().__init__()
        self.res = timm.create_model("resnet34")
        self.res.fc = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.3)
        )

        self.mask_out = self.make_out_layer(3)
        self.gender_out = self.make_out_layer(2)
        self.age_out = self.make_out_layer(3)

        # weight init

    def forward(self, x):
        x = self.res(x)
        mask_out = self.mask_out(x)
        gender_out = self.gender_out(x)
        age_out = self.age_out(x)
        return mask_out, gender_out, age_out

    def make_out_layer(self, num_class):
        return nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_class),
        )


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
        # self.token_embedding = model.token_embedding
        # self.positional_embedding = model.positional_embedding
        # self.transformer = model.transformer
        # self.ln_final = model.ln_final
        # self.text_projection = model.text_projection

        # freeze
        self.is_train(self.encode_image, _train=False)
        # self.is_train(self.token_embedding, _train=True)
        # self.is_train(self.transformer, _train=True)
        # self.is_train(self.ln_final, _train=True)
        # self.positional_embedding.requires_grad_(True)
        # self.text_projection.requires_grad_(True)
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

        self.mask_feature = model.encode_text(
            clip.tokenize(self.mask_text).to(self.device)
        )
        self.gender_feature = model.encode_text(
            clip.tokenize(self.gender_text).to(self.device)
        )
        self.age_feature = model.encode_text(
            clip.tokenize(self.age_text).to(self.device)
        )
        self.cossim = nn.CosineSimilarity(dim=-1)

        self.mask_fc = nn.Linear(1024, 3).type(self.dtype)
        self.gender_fc = nn.Linear(1024, 2).type(self.dtype)
        self.age_fc = nn.Linear(1024, 3).type(self.dtype)

    def forward(self, x):
        img_feature = self.encode_image(x)

        # mask_feature = self.encode_text(self.mask_token).to(self.device)
        # gender_feature = self.encode_text(self.gender_token).to(self.device)
        # age_feature = self.encode_text(self.age_token).to(self.device)

        # mask_out = self.cossim(img_feature, mask_feature)
        # gender_out = self.cossim(img_feature, gender_feature)
        # age_out = self.cossim(img_feature, age_feature)

        mask_img = self.mask_fc(img_feature)
        gender_img = self.gender_fc(img_feature)
        age_img = self.age_fc(img_feature)

        return (mask_img, gender_img, age_img)

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


class Vit_GH(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_base_resnet50_384", pretrained=True)
        self.backbone.blocks = self.backbone.blocks
        for m in self.backbone.parameters():
            m.requires_grad_(False)

        self.backbone.head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.3)
        )

        self.mask_out = self.make_out_layer(3)
        self.gender_out = self.make_out_layer(2)
        self.age_out = self.make_out_layer(3)

        self.layer_list = [
            [
                self.mask_out,
                self.gender_out,
                self.age_out,
                self.backbone.head,
            ],
            *[[block] for block in self.backbone.blocks[::-1]],
            [self.backbone.patch_embed, self.backbone.pos_drop, self.backbone.norm_pre],
        ]
        # weight init
        for layer in self.layer_list[0]:
            for j in layer.parameters():
                if j.dim() == 2:
                    nn.init.kaiming_uniform_(j, nonlinearity="leaky_relu")
                elif j.dim() == 1:
                    j.data.fill_(0.0)

    def forward(self, x):
        x = self.backbone(x)
        mask_out = self.mask_out(x)
        gender_out = self.gender_out(x)
        age_out = self.age_out(x)
        return mask_out, gender_out, age_out

    def make_out_layer(self, num_class):
        return nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_class),
        )

    def is_train(self, modules, _train=True):
        for module in modules:
            for m in module.parameters():
                m.requires_grad_(_train)

    def train_layer(self, epoch):
        self.is_train(self.layer_list[(epoch - 1) % len(self.layer_list)], True)
        self.is_train(self.layer_list[(epoch - 2) % len(self.layer_list)], False)
