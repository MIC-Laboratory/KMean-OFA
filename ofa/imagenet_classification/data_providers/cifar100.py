# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

__all__ = ["Cifar100DataProvider"]


class Cifar100DataProvider(DataProvider):
    DEFAULT_PATH = "/home/zhenyulin/Training_data"

    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=None,
        n_worker=4,
        resize_scale=0.08,
        distort_color=None,
        image_size=32,
        num_replicas=None,
        rank=None,
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            from ofa.utils.my_dataloader.my_data_loader import MyDataLoader

            assert isinstance(self.image_size, list)
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(
                    img_size
                )
            self.active_img_size = max(self.image_size)  # active resolution for test
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform(self.active_img_size)
            train_loader_class = torch.utils.data.DataLoader

        train_dataset = self.train_dataset(self.build_train_transform())

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            train_indexes, valid_indexes = self.random_sample_valid_set(
                len(train_dataset), valid_size
            )

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(
                    train_dataset, num_replicas, rank, True, np.array(train_indexes)
                )

            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    train_indexes
                )


            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )

        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas, rank
                )
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    sampler=train_sampler,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            else:
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=n_worker,
                    pin_memory=True,
                )


        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )

        

    @staticmethod
    def name():
        return "cifar100"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/Training_data")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        return datasets.CIFAR100(self.train_path,train=True,transform=_transforms)

    def test_dataset(self, _transforms):
        return datasets.CIFAR100(self.valid_path,train=False,transform=_transforms)

    @property
    def train_path(self):
        return os.path.join(self.save_path)

    @property
    def valid_path(self):
        return os.path.join(self.save_path)

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.active_img_size
        crop = transforms.RandomCrop((image_size,image_size),padding=4),
            
        train_transforms = transforms.Compose(
            [
            *crop,
            transforms.RandomHorizontalFlip(),
            # transforms.autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            self.normalize
            ])
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        crop = transforms.RandomCrop((image_size,image_size),padding=4),
        return transforms.Compose([
            *crop,
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid and test set
        
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    