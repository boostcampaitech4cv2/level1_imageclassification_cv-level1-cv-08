import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
import torch
import io
from PIL import Image
import numpy as np


def train_transform(input_size):
    return Compose(
        [
            A.Resize(input_size, input_size),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=0.7),
                    A.MedianBlur(blur_limit=3, p=0.7),
                    A.GaussianBlur(blur_limit=3, p=0.7),
                    A.GaussNoise(var_limit=(3.0, 9.0), p=0.7),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=10,
                max_height=20,
                max_width=20,
                min_holes=1,
                min_height=3,
                min_width=3,
                p=0.3,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def test_transform(input_size):
    return Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def streamlit_transform(image_bytes: bytes) -> torch.Tensor:
    transform = Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    return transform(image=image_array)["image"].unsqueeze(0)
