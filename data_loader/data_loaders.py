from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.data_set import TrainDataset, TestDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class TrainLoader(BaseDataLoader):
    """ """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
    ):
        trsfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = TrainDataset(self.data_dir, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class TestLoader(BaseDataLoader):
    """ """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
    ):
        trsfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = TestDataset(self.data_dir, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class TrainLoader_vit(BaseDataLoader):
    """ """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
    ):
        trsfm = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = TrainDataset(self.data_dir, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class TestLoader_vit(BaseDataLoader):
    """ """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
    ):
        trsfm = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = TestDataset(self.data_dir, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
