from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import torch
import numpy as np

# class L1Dataset(Dataset):
#     def __init__(self, df, transform=None, classes=None):
#         self.df, self.transform, self.classes = df, transform, classes
#     def __getitem__(self, idx):
#         data = self.df.iloc[idx]
#         # image = Image.open(data['img_path'])
#         # print(data['img_path']+'/'+data['filename'])
#         try:
#             image = cv2.imread(data['img_path']+'/'+data['filename'])
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         except:
#             print("ERROR:", data['img_path']+'/'+data['filename'], type(image))
#         # if self.transform: image = self.transform(image=image)["image"]
#         if self.transform: image = self.transform(image)

#         # return image, data['mask']
#         multi_class_label = data['mask'] * 6 + data['gender'] * 3 + data['age']
#         return image, multi_class_label

#         # rtn = (data['gender'], int(data['age']), data['mask'])
#         # return image, torch.from_numpy(np.asarray(rtn)).float()

#     def __len__(self): return len(self.df)

class L1Dataset_kind(Dataset):
    def __init__(self, kind, df, transform=None, classes=None):
        self.kind, self.df, self.transform, self.classes = kind, df, transform, classes
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = cv2.imread(data['img_path']+'/'+data['filename'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        # print(self.kind, data[self.kind])
        ##kind 값 'all' 시 멀티클래스인코딩
        if self.kind == 'all': return image, data['mask'] * 6 + data['gender'] * 3 + data['age']
        else: return image, data[self.kind]
        # else: return image, np.asarray(data[self.kind])

    def __len__(self): return len(self.df)

class L1DataLoader_kind(BaseDataLoader):
    def __init__(self, kind, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),              
            transforms.RandomAffine(0, shear=5, scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
        self.data_dir = data_dir

        df = pd.read_csv(f'{self.data_dir}/train.csv')
        dfms = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
        arr = {'incorrect_mask':1, 'mask1':0, 'mask2':0, 'mask3':0, 'mask4':0, 'mask5':0, 'normal':2}

        for key, val in arr.items():
            temp = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
            map_gender = {'male':0, 'female':1}
            temp = df[['gender', 'age']].copy()
            temp.loc[:,('gender')] = df[['gender']].applymap(map_gender.get)
            temp.loc[:,('mask')] = val
            temp.loc[:,('img_path')] = self.data_dir +'images/'+df['path']
            temp.loc[:,('filename')] = key

            ##kind 값 'all' 시 연령값 인코딩
            # if kind == 'all':
            temp.loc[:,('age')] =   np.where(df['age']<30, 0,
                                    np.where(df['age']<60, 1, 2))

            dfms = pd.concat([dfms, temp])
        
        for idx, path in enumerate(dfms['img_path']):
            filename = dfms['filename'].iloc[idx]
            dfms['filename'].iloc[idx] += '.'+self.get_ext(path, filename)

        df_classes = None

        self.dataset = L1Dataset_kind(kind, dfms, transform=trsfm, classes=df_classes)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_ext(self, path, filename):
        with os.scandir(f'{path}/') as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file():
                    name, ext = entry.name.split('.')
                    if filename == name: return ext

# class ToSharp:
#     def __call__(self, src, alpha=1.0, blur=4, norm=True):
#         img_np = np.array(src)/255.
#         blur_img = cv2.GaussianBlur(img_np, (0, 0), blur)
#         sharp_img = (1.0+alpha)*img_np - alpha*blur_img
#         if norm: return cv2.normalize(sharp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#         else: return sharp_img


# class L1DataLoader_kind_sharp_norm(BaseDataLoader):
#     def __init__(self, kind, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             ToSharp(),
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(),              
#             transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#             # transforms.Normalize((0.1307,), (0.3081,)),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
#         ])
#         self.data_dir = data_dir

#         df = pd.read_csv(f'{self.data_dir}/train.csv')
#         dfms = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
#         arr = {'incorrect_mask':1, 'mask1':0, 'mask2':0, 'mask3':0, 'mask4':0, 'mask5':0, 'normal':2}

#         for key, val in arr.items():
#             temp = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
#             map_gender = {'male':0, 'female':1}
#             temp = df[['gender', 'age']].copy()
#             temp.loc[:,('gender')] = df[['gender']].applymap(map_gender.get)
#             temp.loc[:,('mask')] = val
#             temp.loc[:,('img_path')] = self.data_dir +'images/'+df['path']
#             temp.loc[:,('filename')] = key

#             ##kind 값 'all' 시 연령값 인코딩
#             # if kind == 'all':
#             temp.loc[:,('age')] =   np.where(df['age']<30, 0,
#                                     np.where(df['age']<60, 1, 2))

#             dfms = pd.concat([dfms, temp])
        
#         for idx, path in enumerate(dfms['img_path']):
#             filename = dfms['filename'].iloc[idx]
#             dfms['filename'].iloc[idx] += '.'+self.get_ext(path, filename)

#         df_classes = None

#         self.dataset = L1Dataset_kind(kind, dfms, transform=trsfm, classes=df_classes)

#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#     def get_ext(self, path, filename):
#         with os.scandir(f'{path}/') as it:
#             for entry in it:
#                 if not entry.name.startswith('.') and entry.is_file():
#                     name, ext = entry.name.split('.')
#                     if filename == name: return ext
