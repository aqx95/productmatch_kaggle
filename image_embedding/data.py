import cv2
import os
import torch
from pandas as pd
from torch.utils.data import Dataset, DataLoader

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Rotate,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


class ShopeeDataset(Dataset):
    def __init__(self, df, config, transforms=None):
        self.df = df.reset_index()
        self.config = config
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        image_path = os.path.join(self.config.paths['TRAIN_PATH'], self.df.loc[index]['image_id'])
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        labels = torch.tensor(row.label_group)

        return image, labels


def get_train_transforms(config):
    return Compose(
        [
            Resize(config.img_size, config.img_size, always_apply=True),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(p=0.5),
            RandomBrightness(limit=(0.09, 0.6), p=0.5),
            #albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            #albumentations.ShiftScaleRotate(
              #  shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            #),
            Normalize(),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms(config):
    return Compose(
        [
            Resize(config.img_size, config.img_size, always_apply=True),
            Normalize(),
        ToTensorV2(p=1.0)
        ]
    )


def prepare_loader(train_df, valid_df, config):
    train_ds = ShopeeDataset(train_df, config, transforms=get_train_transforms(config))
    valid_ds = ShopeeDataset(valid_df, config, transforms=get_valid_transforms(config))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2)

    return train_loader, valid_loader
