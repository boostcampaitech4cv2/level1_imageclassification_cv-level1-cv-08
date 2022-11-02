import re
from glob import glob
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from data_loader.transforms import train_transform, test_transform


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
    labels = []
    for file in files:
        name, mask = re.search("_(.+?)[.]", file)[1].split("/")
        mask = re.sub("[0-9]", "", mask)
        gender, _, age = name.split("_")

        if mask == "mask":
            mask = 0
        elif mask == "incorrect_mask":
            mask = 1
        elif mask == "normal":
            mask = 2
        else:
            raise ValueError(f"Invalid mask type {mask}")

        if gender == "male":
            gender = 0
        elif gender == "female":
            gender = 1
        else:
            raise ValueError(f"Invalid gender type {gender}")

        if (age := int(age)) < 20:
            age = 0
        elif age < 30:
            age = 1
        elif age < 40:
            age = 2
        elif age < 50:
            age = 3
        elif age < 60:
            age = 4
        elif age >= 60:
            age = 5
        else:
            raise ValueError(f"Invalid age type {age}")
        ages = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
        label = mask * 6 + gender * 3 + ages[age]

        labels.append([label, mask, gender, age])
    return list(zip(files, labels))


def make_dataset(stage="train"):
    path = f"/opt/ml/input/data/{stage}"
    if stage == "train":
        path = path + "/images"
        files = glob(f"{path}/*/*.jpg")
        files += glob(f"{path}/*/*.jpeg")
        files += glob(f"{path}/*/*.png")
        assert len(files) == 18900, "All dataset is not complete"
        files = prepare_dataset(files)

        train_set, valid_set = train_test_split(
            files, test_size=0.1, random_state=42, shuffle=True
        )
        print(f"train dataset size : {len(train_set)}")
        print(f"valid dataset size : {len(valid_set)}")

        return train_set, valid_set
    if stage == "eval":
        submission = pd.read_csv(os.path.join(path, "info.csv"))
        image_dir = os.path.join(path, "images")
        image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

        assert len(image_paths) == 12600, "Whole dataset is not complete"

        return image_paths


def setup(
    stage="train",
    input_size=224,
    batch_size=32,
    num_workers=4,
):

    if stage == "train":
        print("train dataset loading")
        train_set, valid_set = make_dataset(stage)

        train_set = CustomDataset(train_set, transform=train_transform(input_size))
        valid_set = CustomDataset(valid_set, transform=test_transform(input_size))

        train_dataloader = DataLoader(
            train_set,
            batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            drop_last=True,
        )
        valid_dataloader = DataLoader(
            valid_set,
            batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            drop_last=True,
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
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    setup(stage="train")
