import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainDataset(Dataset):
    def __init__(
        self,
        path="/opt/ml/level1_imageclassification_cv-level1-cv-08/data_loader/data/train/",
        transform=None,
    ):
        """
        mask : (wear, 0), (incorrect, 1), (normal, 2)
        gender : (male ,0), (female, 1)
        age : (0, <30), (1, >= 30 and < 60), (2, <= 60)
        """
        train_csv = pd.read_csv(path + "train.csv").path
        train_csv = pd.DataFrame(
            [
                f"{j}/{i}"
                for j in train_csv
                for i in os.listdir(path + "images/" + j)
                if i[0] != "."
            ],
            columns=["path"],
        )
        train_csv = pd.concat(
            [
                train_csv,
                pd.DataFrame(
                    [i.split("_") + [j] for i, j in train_csv.path.str.split("/")],
                    columns=["id", "gender", "race", "age", "mask"],
                ),
            ],
            axis=1,
        )
        train_csv["age"] = pd.cut(
            train_csv.age.astype(int), [0, 30, 60, 255], right=False, labels=[0, 1, 2]
        )
        train_csv["gender"] = train_csv.gender.map({"male": 0, "female": 1})
        train_csv["mask"] = train_csv["mask"].map(
            lambda x: 1 if "incorrect" in x else (2 if "normal" in x else 0)
        )

        self.train_csv = train_csv[["path", "id", "gender", "age", "mask"]]
        self.data = [Image.open(path + "images/" + path2) for path2 in train_csv.path]
        self.target = [
            np.array([mask, gender, age])
            for mask, gender, age in train_csv[["mask", "gender", "age"]].iloc
        ]

        self.transform = transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            target : (mask ,gender age)
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.target[index]

    def __len__(self) -> int:
        return len(self.train_csv)


class TestDataset(Dataset):
    def __init__(
        self,
        path="/opt/ml/level1_imageclassification_cv-level1-cv-08/data_loader/data/eval/",
        transform=None,
    ):
        """
        mask : (wear, 0), (incorrect, 1), (normal, 2)
        gender : (male ,0), (female, 1)
        age : (0, <30), (1, >= 30 and < 60), (2, <= 60)
        """
        self.info_csv = pd.read_csv(path + "info.csv")
        self.data = [
            Image.open(path + "images/" + path2) for path2 in self.info_csv.ImageID
        ]
        self.transform = transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            target : (mask ,gender age)
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.info_csv)
