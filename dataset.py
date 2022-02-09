import os
from typing import Tuple, Optional

from PIL import Image
import numpy as np
import cv2
import random
import torch

from pytorch_lightning import LightningDataModule
from torch.utils import data
from torchvision import transforms

BASE_PATH = "./mvtec-ad"

TRAIN = "train"
TEST = "test"
VALIDATE = "validate"


class MVTecAD(data.Dataset):
    def __init__(self, image_list, label_list, transform, stage='train'):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.image_list) * (1 if self.stage == 'test' else 600)

    def __getitem__(self, index):
        image_index = index % len(self.image_list)
        image = Image.open(self.image_list[image_index]).convert("RGB")
        label = self.label_list[image_index]

        return self.transform(image), label


class MVTecADDataModule(LightningDataModule):
    test_dataset = []
    train_dataset = []
    val_dataset = []

    def __init__(
        self,
        image_type: str,
        image_size: int,
        patch_size: int,
        window_size: int,
        train_ratio: float = 0.9,
        batch_size: int = 32,
        num_workers: int = os.cpu_count(),
        seed: int = 42,
    ):
        super().__init__()
        self.image_type = image_type
        self.patch_size = patch_size
        self.window_size = window_size
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self) -> None:
        super().prepare_data()
        image_dir = os.path.join(BASE_PATH, self.image_type)
        test_imgdir = os.path.join(image_dir, "test")
        test_labdir = os.path.join(image_dir, "ground_truth")

        test_image_list = self._get_image_list(test_imgdir)
        test_mask_list = [
            self._get_image_mask(test_imgdir, test_labdir, x) for x in test_image_list
        ]

        self.test_dataset = MVTecAD(
            test_image_list,
            test_mask_list,
            self._transform_infer(),
            stage='test'
        )

        train_imgdir = os.path.join(image_dir, os.path.join("train", "good"))
        train_image_list = self._get_image_list(train_imgdir)
        random.shuffle(train_image_list)

        train_size = len(train_image_list) - min(
            20, int(len(train_image_list) * (1 - self.train_ratio))
        )

        val_image_list = train_image_list[train_size:]
        val_mask_list = [
            (np.zeros((self.image_size, self.image_size), dtype=np.uint8), 0)
        ] * len(val_image_list)

        print("Amount of val images in dataset: ", len(val_image_list))
        print("Amount of val masks in dataset: ", len(val_mask_list))

        train_image_list = train_image_list[:train_size]
        train_mask_list = [
            (np.zeros((self.image_size, self.image_size), dtype=np.uint8), 0)
        ] * len(train_image_list)

        print("Amount of train images in dataset: ", len(train_image_list))
        print("Amount of train masks in dataset: ", len(train_mask_list))

        self.train_dataset = MVTecAD(
            train_image_list,
            train_mask_list,
            self._transform_train(),
            stage='train'
        )
        self.val_dataset = MVTecAD(
            val_image_list,
            val_mask_list,
            self._transform_infer(),
            stage='val'
        )

        print("Number of train patches in dataset: ", len(self.train_dataset))
        print("Number of val patches in dataset: ", len(self.train_dataset))

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...

    def _transform_train(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
            ]
        )

    def _transform_infer(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def _get_image_list(self, path: str):
        image_list = []
        for (root, dirs, files) in os.walk(path):
            for fname in files:
                if os.path.splitext(os.path.basename(fname))[-1] in [
                    ".jpeg",
                    ".jpg",
                    ".bmp",
                    ".png",
                    ".tif",
                ]:
                    image_list.append(os.path.join(root, fname))
        return image_list

    def _get_image_mask(self, test_imgdir: str, truth_imgdir: str, test_imgpath: str):
        """

        :param test_imgdir: Directory of the test images
        :param truth_imgdir: Directory where the ground truth files are stored
        :param test_imgpath: Full path to the current image in the test directory
        :param img_size: Size of the image
        :return:
        """
        # test_imgdir : bottle/test
        # truth_imgdir : bottle/ground_truth
        # test_imgpath : bottle/test/broken_large/000.png ||| bottle/test/good/000.png
        # test_labpath : bottle/ground_truth/broken_large/000_mask.png
        truth_imgpath_base = test_imgpath.replace(test_imgdir, truth_imgdir)
        truth_imagepath = os.path.join(
            truth_imgpath_base.split(".png")[0] + "_mask.png"
        )
        if os.path.exists(truth_imagepath):
            mask = cv2.resize(
                cv2.imread(truth_imagepath, cv2.IMREAD_GRAYSCALE),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            mask[mask >= 1] = 1
            return mask, 1
        else:
            print("Mask does not exist: ", test_imgpath)
            return np.zeros(shape=(self.image_size, self.image_size)), 0


def _imshow(x_0):
    print(len(x_0))
    print(x_0.keys())
    print(x_0["target_position"])
    x_0 = x_0["target_patch"]
    for i in range(list(x_0.size())[0]):
        img = x_0[i].detach().cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    image_type = "bottle"
    image_size = 256
    data_module = MVTecADDataModule(
        num_workers=1,
        patch_size=16,
        window_size=7,
        image_type=image_type,
        image_size=image_size,
        train_ratio=1.0,
    )
    data_module.setup()
    data_module.prepare_data()
    data_loader = data_module.train_dataloader()
    x_0, _ = iter(data_loader).next()
    _imshow(x_0)
