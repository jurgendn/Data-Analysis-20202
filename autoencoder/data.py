from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision import transforms


class CIFAR10Data(CIFAR10DataModule):
    def __init__(self,
                data_dir: str = None,
                num_workers: int = 4,
                batch_size: int = 128,
                shuffle: bool = False):
        super(CIFAR10Data, self).__init__(
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            train_transforms=self.train_transforms,
            val_transforms=self.test_transforms,
            test_transforms=self.test_transforms
        )

    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization(),
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            cifar10_normalization(),
        ])

    @property
    def unnormalization(self):
        return transforms.Normalize(
            mean=[- x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[- x / 255.0 for x in [63.0, 62.1, 66.7]],
        )