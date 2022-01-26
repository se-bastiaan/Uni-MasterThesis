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
    def __init__(self, patch_size, window_size, image_list, label_list, transform):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.patch_size = patch_size
        self.window_size = window_size

    def __len__(self):
        return len(self.image_list) * 600

    def __getitem__(self, index):
        image_index = index % len(self.image_list)
        image = Image.open(self.image_list[image_index]).convert("RGB")
        label = self.label_list[image_index]

        X = self.transform(image)

        # sampled random window from image
        _, H, W = X.shape
        assert (H % self.patch_size == 0) and (
            W % self.patch_size == 0
        ), "Expected image width and height are divisible by patch size."

        patches_info: dict = self._to_window_patches(X)

        return patches_info, label

    def _to_window_patches(
        self,
        x
    ):
        with torch.no_grad():
            C, H, W = x.shape
            vertical_patch_count = H // self.patch_size
            horizontal_patch_count = W // self.patch_size

            window_x, window_y = (
                np.round(
                    np.random.uniform(0, vertical_patch_count - self.window_size - 1)
                ).astype(int),
                np.round(
                    np.random.uniform(0, horizontal_patch_count - self.window_size - 1)
                ).astype(int),
            )
            patch_x, patch_y = (
                np.round(np.random.uniform(0, self.window_size - 1)).astype(int),
                np.round(np.random.uniform(0, self.window_size - 1)).astype(int),
            )

            # Reshape [C, H, W] --> [C, N, patch_size, M, patch_size]
            x = x.reshape(
                C,
                vertical_patch_count,
                self.patch_size,
                horizontal_patch_count,
                self.patch_size,
            )
            # Re-arrange axis [C, N, patch_size, M, patch_size] --> [N, M, C, patch_size, patch_size]
            #                 [0, 1, 2, 3, 4] --> [1, 3, 0, 2, 4]
            x = x.permute(1, 3, 0, 2, 4)

            # We will choose window_size*window_size sub-grid from M*N gird.
            # The coordinate(index) of top-left patch of window_size*window_size grid will be denoted as (r, s).
            # Uniform random integer will be generated for r, s coordinate(index)
            # slicing [N, M, C, patch_size, patch_size] --> [window_size, window_size, C, patch_size, patch_size]
            window_subgrid = x[
                window_x : window_x + self.window_size,
                window_y : window_y + self.window_size,
                :,
            ]

            # Positional encoding for above selected window_size*window_size patches.
            # We will use global index where the top-left patch is 0
            # and follow left-to-right english writing style.
            grid_positions = torch.arange(
                0, vertical_patch_count * horizontal_patch_count, dtype=torch.long
            ).reshape(vertical_patch_count, horizontal_patch_count)
            subgrid_positions = grid_positions[
                window_x : window_x + self.window_size,
                window_y : window_y + self.window_size,
            ]

            # Flatten in window_size dimension
            # [window_size, window_size, C, patch_size, patch_size] --> [(window_size, window_size), C, patch_size, patch_size]
            # i.e.: [7, 7, C, patch_size, patch_size] --> [49, C, patch_size, patch_size]
            window_subgrid = window_subgrid.flatten(0, 1)
            subgrid_positions = subgrid_positions.flatten(0)  # i.e. [7, 7] -> [49]

            # we will take one patch from window_size*window_size patches which we will try to inpaint.
            # For example, if window_size=7, then we will randomly take 1 patch as mask from 49(7*7).
            # Remaining 48 will be feed into model. The goal is to predict the unknown one.
            patch_id = (patch_x * self.window_size) + patch_y
            # Choose target inpaint patch [1, C, patch_size, patch_size]
            # i.e.: [1, C, patch_size, patch_size]
            target_patch = window_subgrid[patch_id, :, :, :]
            target_subgrid_position = subgrid_positions[patch_id]
            # Separate remaining patches. These will be the conditioning neighbors.
            # i.e.: [48, C, patch_size, patch_size]
            context_patches = torch.cat(
                [
                    window_subgrid[0:patch_id, :, :, :],
                    window_subgrid[patch_id + 1 :, :, :, :],
                ],
                0,
            )

            context_subgrid_positions = torch.cat(
                [subgrid_positions[:patch_id], subgrid_positions[patch_id + 1 :]], 0
            )

        return {
            "context_patches": context_patches.flatten(1, 3),
            "context_positions": context_subgrid_positions,
            "target_patch": target_patch,
            "target_position": target_subgrid_position,
        }


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
        num_workers: int = 4,
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
        random.seed(self.seed)
        image_dir = os.path.join(BASE_PATH, self.image_type)
        test_imgdir = os.path.join(image_dir, "test")
        test_labdir = os.path.join(image_dir, "ground_truth")

        test_image_list = self._get_image_list(test_imgdir)
        test_mask_list = [
            self._get_image_mask(test_imgdir, test_labdir, x) for x in test_image_list
        ]

        self.test_dataset = MVTecAD(
            self.patch_size,
            self.window_size,
            test_image_list,
            test_mask_list,
            self._transform_infer(),
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
            self.patch_size,
            self.window_size,
            train_image_list,
            train_mask_list,
            self._transform_train(),
        )
        self.val_dataset = MVTecAD(
            self.patch_size,
            self.window_size,
            val_image_list,
            val_mask_list,
            self._transform_infer(),
        )

        print("Number of train patches in dataset: ", len(self.train_dataset))
        print("Number of val patches in dataset: ", len(self.train_dataset))

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
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
