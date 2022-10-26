import re
from glob import glob
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

import cv2
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, dataset, test=False, transform=None):
        self.pair_list = dataset
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        image = cv2.imread(pair) if self.test else cv2.imread(pair[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)

        return image["image"] if self.test else (image["image"], pair[1])


def prepare_dataset(files):
    classes = {
        "male_y/mask": 0,
        "male_m/mask": 1,
        "male_o/mask": 2,
        "female_y/mask": 3,
        "female_m/mask": 4,
        "female_o/mask": 5,
        "male_y/incorrect_mask": 6,
        "male_m/incorrect_mask": 7,
        "male_o/incorrect_mask": 8,
        "female_y/incorrect_mask": 9,
        "female_m/incorrect_mask": 10,
        "female_o/incorrect_mask": 11,
        "male_y/normal": 12,
        "male_m/normal": 13,
        "male_o/normal": 14,
        "female_y/normal": 15,
        "female_m/normal": 16,
        "female_o/normal": 17,
    }
    labels = []
    for file in files:
        name, mask = re.search("_(.+?)[.]", file)[1].split("/")
        mask = re.sub("[0-9]", "", mask)
        sex, _, age = name.split("_")
        if int(age) < 30:
            age = "y"
        elif int(age) < 60:
            age = "m"
        else:
            age = "o"
        label = classes[f"{sex}_{age}/{mask}"]
        labels.append(label)
    return list(zip(files, labels))


def make_dataset(stage="train"):
    path = f"/opt/ml/data/{stage}/images"
    if stage == "train":
        files = glob(f"{path}/*/*.jpg")
        train, valid = train_test_split(
            files, test_size=0.2, random_state=42, shuffle=True
        )
        train_set = prepare_dataset(train)
        valid_set = prepare_dataset(valid)
        return train_set, valid_set
    if stage == "eval":
        files = glob(f"{path}/*.jpg")
        return files


def setup(
    stage="train",
    input_size=224,
    batch_size=32,
    num_workers=0,
):
    train_transform = Compose(
        [
            A.CenterCrop(356, 356),
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    test_transform = Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    if stage == "train":
        print("train dataset loading")
        train_set, valid_set = make_dataset(stage)

        train_set = CustomDataset(train_set, transform=train_transform)
        valid_set = CustomDataset(valid_set, transform=test_transform)

        train_dataloader = DataLoader(
            train_set, batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_set,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        return train_dataloader, valid_dataloader
    if stage == "eval":
        print("eval dataset loading")
        test_set = make_dataset(stage)
        test_set = CustomDataset(test_set, test=True, transform=test_transform)
        return DataLoader(
            test_set,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )


if __name__ == "__main__":
    setup(stage="train")
