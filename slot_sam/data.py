import glob
import json
import os
from typing import Callable
from typing import List
from typing import Optional

import cv2
import numpy as np
# import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from mobile_sam.utils.transforms import ResizeLongestSide
# from slot_sam.utils import compact


class Shapes2dDataset(Dataset):
    def __init__(
        self, path: str, transform: Callable
    ):
        super().__init__()
        self.path = path
        self.transform = transform
        self.files = glob.glob(os.path.join(self.path, '*'))

    def __getitem__(self, index: int):
        image_path = self.files[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.transform(image), image

    def __len__(self):
        return len(self.files)


class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "images", split)
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(self, index: int):
        image_path = self.files[index]
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.clevr_transforms(img)

    def __len__(self):
        return len(self.files)

    def get_files(self) -> List[str]:
        with open(os.path.join(self.data_root, f"scenes/CLEVR_{self.split}_scenes.json")) as f:
            scene = json.load(f)
        paths: List[Optional[str]] = []
        total_num_images = len(scene["scenes"])
        i = 0
        while (self.max_num_images is None or len(paths) < self.max_num_images) and i < total_num_images:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            if num_objects_in_scene <= self.max_n_objects:
                image_path = os.path.join(self.data_path, scene["scenes"][i]["image_filename"])
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)
            i += 1
        return sorted(compact(paths))


# class CLEVRDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         data_root: str,
#         train_batch_size: int,
#         val_batch_size: int,
#         clevr_transforms: Callable,
#         max_n_objects: int,
#         num_workers: int,
#         num_train_images: Optional[int] = None,
#         num_val_images: Optional[int] = None,
#     ):
#         super().__init__()
#         self.data_root = data_root
#         self.train_batch_size = train_batch_size
#         self.val_batch_size = val_batch_size
#         self.clevr_transforms = clevr_transforms
#         self.max_n_objects = max_n_objects
#         self.num_workers = num_workers
#         self.num_train_images = num_train_images
#         self.num_val_images = num_val_images
#
#         self.train_dataset = CLEVRDataset(
#             data_root=self.data_root,
#             max_num_images=self.num_train_images,
#             clevr_transforms=self.clevr_transforms,
#             split="train",
#             max_n_objects=self.max_n_objects,
#         )
#         self.val_dataset = CLEVRDataset(
#             data_root=self.data_root,
#             max_num_images=self.num_val_images,
#             clevr_transforms=self.clevr_transforms,
#             split="val",
#             max_n_objects=self.max_n_objects,
#         )
#
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.train_batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=False,
#         )
#
#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.val_batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=False,
#         )


class ResizeSam:
    def __init__(self, target_length):
        super().__init__()
        self.resize_longest_side = ResizeLongestSide(target_length=target_length)

    def __call__(self, image: np.ndarray):
        return self.resize_longest_side.apply_image(image)


class PreprocessSam:
    def __init__(self, pixel_mean, pixel_std):
        super().__init__()
        self.pixel_mean = torch.as_tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        self.pixel_std = torch.as_tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: np.ndarray):
        image_torch = torch.as_tensor(image)
        image_torch = image_torch.permute(2, 0, 1)

        image_torch = (image_torch - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = image_torch.shape[-2:]
        img_size = max(h, w)
        padh = img_size - h
        padw = img_size - w
        image_torch = torch.nn.functional.pad(image_torch, (0, padw, 0, padh))
        return image_torch


class TransformSam:
    def __init__(self, target_length=1024, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        self.transforms = transforms.Compose(
            [
                ResizeSam(target_length),
                PreprocessSam(pixel_mean, pixel_std)
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
