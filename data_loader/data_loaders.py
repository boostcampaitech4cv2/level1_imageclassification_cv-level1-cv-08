from torchvision import datasets, transforms
from . import dataset as ds
from torch.utils.data import DataLoader
from . import transformer as T

# from base import BaseDataLoader


class MaskDataLoader(DataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        num_workers=1,
        training=True,
    ):
        self.transform = T.BasicTransform()
        # transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.training = training
        self.dataset = ds(
            data_dir=self.data_dir, train=self.training, transform=self.transform
        )
        super().__init__(
            data_dir=self.data_dir,
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
