#!pip install timm

import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# class ResNet50Model(BaseModel):
#     def __init__(self, num_classes=18):
#         super().__init__()
#         self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

#     def forward(self, x):
#         return self.model(x)

# class EfficentModel(BaseModel):
#     def __init__(self, num_classes=18):
#         super().__init__()
#         self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)

#     def forward(self, x):
#         return self.model(x)

class TimmModel(BaseModel):
    def __init__(self, model_name='efficientnet_b4', num_classes=18):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)