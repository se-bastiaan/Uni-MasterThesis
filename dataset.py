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

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        label = self.label_list[index]

        X = self.transform(image)

        # sampled random window from image
        K, L = self.patch_size, self.window_size
        C, H, W = X.shape
        N = H // K
        M = W // K
        with torch.no_grad():
            # (r, s) is the window's top-left patch coordinate
            r = (
                torch.FloatTensor(1)
                .uniform_(0, N - L - 1)
                .round()
                .type(torch.long)
                .item()
            )
            s = (
                torch.FloatTensor(1)
                .uniform_(0, M - L - 1)
                .round()
                .type(torch.long)
                .item()
            )
            # (t, u) is the inpainted patch coordinate
            t = torch.FloatTensor(1).uniform_(0, L - 1).round().type(torch.long).item()
            u = torch.FloatTensor(1).uniform_(0, L - 1).round().type(torch.long).item()

        patches_info: dict = self._to_window_patches(
            X, K=K, L=L, r=r, s=s, t=t, u=u, flatten_patch=True
        )

        return patches_info, label

    @staticmethod
    def _to_window_patches(x, K, L, r, s, t, u, flatten_patch=True):
        """
        Source: https://github.com/uzl/inpainting-transformer/blob/master/inpainting_transformer_base.py
        It will make K by K fixed patchs from an image.
        Then it will randomly choose square grid of L by L patches.
        One patch from L by L patches will be used as inpainting patch. (This one we want to produce from model)
        Remaining patches will be feed as input of the model.
        An example:
            for image shape (3, 384, 384),
            the main output will be (others positional information is not shown here):
                - torch.Size([48, 768])
                - torch.Size([1, 3, 16, 16]])  # it is not flatten because it will be used as image.
                or, when flatten_patch=False
                - torch.Size([48, 3, 16, 16])
                - torch.Size([1, 3, 16, 16])
        Args:
            x: Image (Channel, Height, Width).
            K (int): Patch size.
                For example, if we want to want to make a 16*16 pixel patch,
                then k=16.
            L (int): Subgrid arm length. We we want to take 7*7 patches from all M*N patches,
                then L=7
            r (int): top position of (r, s) pair of subgrid start patch
            s (int): right position of (r, s) pair of subgrid start patch
            t (int): Local top position of (t, u) pair of inpainting patch
            u (int): Local top position of (t, u) pair of inpaint pingatch

            For more details of (r, s) and (t, u) pairs, please see section-3.1 of the paper.
        """
        with torch.no_grad():
            C, H, W = x.shape
            assert (H % K == 0) and (
                W % K == 0
            ), "Expected image width and height are divisible by patch size."
            N = H // K
            M = W // K

            # Reshape [C, H, W] --> [C, N, K, M, K]
            x = x.reshape(C, N, K, M, K)
            # Re-arrange axis [C, N, K, M, K] --> [N, M, C, K, K]
            #                 [0, 1, 2, 3, 4] --> [1, 3, 0, 2, 4]
            x = x.permute(1, 3, 0, 2, 4)

            # We will choose L*L sub-grid from M*N gird.
            # The coordinate(index) of top-left patch of L*L grid will be denoted as (r, s).
            # Uniform random integer will be generated for r, s coordinate(index)
            # slicing [N, M, C, K, K] --> [L, L, C, K, K]
            sub_x = x[
                r : r + L,
                s : s + L,
                :,
            ]

            # Positional encoding for above selected L*L patchs.
            # We will use global index where the top-left patch is 0
            # and follow left-to-right english writing style.
            all_pos_idx = torch.arange(0, N * M, dtype=torch.long).reshape(N, M)
            sub_pos_idx = all_pos_idx[r : r + L, s : s + L]

            # Flatten in L dimension
            # [L, L, C, K, K] --> [(L, L), C, K, K]
            # i.e.: [7, 7, C, K, K] --> [49, C, K, K]
            sub_x = sub_x.flatten(0, 1)
            sub_pos_idx = sub_pos_idx.flatten(0)  # i.e. [7, 7] -> [49]

            # we will take one patch from L*L patches which we will try to inpaint.
            # For example, if L=7, then we will randomly take 1 patch as mask from 49(7*7).
            # Remaining 48 will be feed into model. The goal is to predict that msked one.
            mask_idx = (u * L) + t
            # Choose target inpaint patch [1, C, K, K]
            # i.e.: [1, C, K, K]
            inpaint_patch = sub_x[mask_idx, :, :, :]
            inpaint_pos_idx = sub_pos_idx[mask_idx]
            # Separate remaining patches. These will be the conditioning neighbors.
            # i.e.: [48, C, K, K]
            neighbor_patchs = torch.cat(
                [sub_x[0:mask_idx, :, :, :], sub_x[mask_idx + 1 :, :, :, :]], 0
            )

            neighbor_pos_idxs = torch.cat(
                [sub_pos_idx[:mask_idx], sub_pos_idx[mask_idx + 1 :]], 0
            )

            if flatten_patch:
                # Flatten in K and C dimensions. For example-
                # in neighbor_patchs: [48, C, K, K] --> [48, (C, K, K)]
                neighbor_patchs = neighbor_patchs.flatten(1, 3)

        return {
            "neighbor_patchs": neighbor_patchs,
            "neighbor_positions": neighbor_pos_idxs,
            "inpaint_patch": inpaint_patch,
            "inpaint_position": inpaint_pos_idx,
        }

    def __len__(self):
        return len(self.image_list)


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
            self._get_image_mask(test_imgdir, test_labdir, x)
            for x in test_image_list
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

        train_image_list = train_image_list[
            : int(len(train_image_list) * self.train_ratio)
        ]
        train_mask_list = [(np.zeros((self.image_size, self.image_size), dtype=np.uint8), 0)] * len(
            train_image_list
        )

        print('Amount of train images in dataset: ', len(train_image_list))
        print('Amount of train masks in dataset: ', len(train_mask_list))

        val_image_list = train_image_list[
            int(len(train_image_list) * self.train_ratio) :
        ]
        val_mask_list = [(np.zeros((self.image_size, self.image_size), dtype=np.uint8), 0)] * len(
            val_image_list
        )

        print('Amount of val images in dataset: ', len(val_image_list))
        print('Amount of val masks in dataset: ', len(val_mask_list))

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

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

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
            [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]
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

    def _get_image_mask(
        self,
        test_imgdir: str,
        truth_imgdir: str,
        test_imgpath: str
    ):
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
    for i in range(list(x_0.size())[0]):
        img = x_0[i].detach().cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    image_type = "bottle"
    image_size = (256, 256)
    data_module = MVTecADDataModule(image_type, image_size, train_ratio=1.0)
    data_module.setup()
    data_loader = data_module.train_dataloader()
    x_0, _ = iter(data_loader).next()
    _imshow(x_0)
