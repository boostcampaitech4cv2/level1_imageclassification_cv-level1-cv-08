import albumentations as A
from albumentations.pytorch import ToTensorV2


def BasicTransform():
    return A.Compose([A.Resize(256, 256), ToTensorV2()])


def StrongTransform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            ToTensorV2(),
        ]
    )
