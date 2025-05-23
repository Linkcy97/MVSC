from torch.utils.data import Dataset, random_split
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
import sys

if sys.gettrace() is not None:
    NUM_DATASET_WORKERS = 1
else:
    NUM_DATASET_WORKERS = 8

SCALE_MIN = 0.75
SCALE_MAX = 0.95


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)

class HR_image(Dataset):
    def __init__(self, config, data_dir, add_noise=False):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.Resize([self.im_height, self.im_width]),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        # transforms.ToPILImage()(transformed).save('test.png')
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        transformed = self.transform(image)
        # transforms.ToPILImage()(transformed).save('test.png')
        return transformed
    def __len__(self):
        return len(self.imgs)

class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)

    def __len__(self):
        return self.len * 10


def get_loader(config):
    if config.datasets == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        val_dataset = Datasets(config.val_data_dir)
        val_dataset = data.Subset(val_dataset, range(20))
        test_dataset = Datasets(config.test_data_dir)
    elif config.datasets == 'CIFAR10':
        dataset_ = datasets.CIFAR10
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
            
        train_dataset = dataset_(root=config.train_data_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=False)
        
        train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])
        test_dataset = dataset_(root=config.val_data_dir,
                                train=False,
                                transform=transform_test,
                                download=False)

        kodak_dataset = Datasets(config.test_data_dir)
        train_dataset = CIFAR10(train_dataset)

    else:
        train_dataset = Datasets(config.train_data_dir)
        val_dataset = Datasets(config.val_data_dir)
        test_dataset = Datasets(config.test_data_dir)


    train_loader = data.DataLoader(dataset=train_dataset,
                                            num_workers=NUM_DATASET_WORKERS,
                                            pin_memory=True,
                                            batch_size=config.batch_size,
                                            worker_init_fn=worker_init_fn_seed,
                                            shuffle=False,
                                            drop_last=True)
    
    if config.datasets == 'CIFAR10':
        val_loader = data.DataLoader(dataset=val_dataset,
                                            batch_size=512,
                                            shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=512,
                                  shuffle=False)
        kodak_loader = data.DataLoader(dataset=kodak_dataset,
                                  batch_size=1,
                                  shuffle=False)
    elif config.datasets == 'DIV2K':
        test_loader = data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    
        val_loader = data.DataLoader(dataset=val_dataset,
                                                batch_size=1,
                                                shuffle=False)
    
    return train_loader, val_loader, test_loader, kodak_loader




