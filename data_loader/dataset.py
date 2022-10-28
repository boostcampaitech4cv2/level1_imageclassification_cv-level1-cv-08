import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import json
from glob import glob

data_dir = "/opt/ml/input/data/train/images"


def get_labels(img_paths):
    tmp_img_paths = img_paths
    labels = []

    for img_path in tmp_img_paths:
        if img_path.split("/")[-1] == "m":
            mask = "Wear"
        elif img_path.split("/")[-1] == "n":
            mask = "Not Wear"
        elif img_path.split("/")[-1] == "i":
            mask = "Incorrect"

        pathInfo = img_path.split("/")[-2].split("_")

        if int(pathInfo[-1]) < 30:
            age = "<30"
        elif int(pathInfo[-1]) < 60:
            age = ">=30 and < 60"
        else:
            age = ">=60"

        gender = pathInfo[-3].title()

        label = {"Mask": mask, "Gender": gender, "Age": age}

        labels.append(label)
    return labels


class MyDataset(Dataset):
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, train, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.img_paths = glob(os.path.join(f"{data_dir}", "**/*"))
        self.make_labels()

    def make_labels(self):
        labels = get_labels(self.img_paths)
        for label in labels:
            self.mask_labels.append(label["Mask"])
            self.gender_labels.append(label["Gender"])
            self.age_labels.append(label["Age"])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        mask = self.mask_labels[idx]
        gender = self.gender_labels[idx]
        age = self.age_labels[idx]

        img_transform = self.transform(np.array(img))
        return img_transform, mask, gender, age
