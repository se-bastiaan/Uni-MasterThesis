import os
from argparse import ArgumentParser
from os import listdir, path
from os.path import isfile, join

import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import mlflow.pytorch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from dataset import MVTecADDataModule
from metrics import (
    compute_pixelwise_retrieval_metrics,
    compute_imagewise_retrieval_metrics,
)
from model import InTra
from utils import tensor2nparr, get_basename

IMAGE_SIZE = {
    "carpet": 512,
    "grid": 256,
    "leather": 512,
    "tile": 512,
    "wood": 512,
    "bottle": 256,
    "cable": 256,
    "capsule": 320,
    "hazelnut": 256,
    "metal_nut": 256,
    "pill": 512,
    "screw": 320,
    "toothbrush": 256,
    "transistor": 256,
    "zipper": 512,
}


def main(args):
    args.image_size = (
        args.image_size if args.image_size else IMAGE_SIZE[args.image_type]
    )

    pl.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Sees device " + str(device))

    tb_logger = pl_loggers.TensorBoardLogger(
        f"{args.output_path}/logs/",
        name=f"{args.image_type}-{args.max_epochs}-{args.attention_type}",
    )
    loggers = [tb_logger]

    checkpoint_best = ModelCheckpoint(
        filename="best-{epoch}-{step}-{val_loss:.5f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    checkpoint_last = ModelCheckpoint(
        filename="last-{epoch}-{step}",
        save_last=True,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=args.patience
    )

    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        checkpoint_path = f"{args.output_path}/{args.image_type}-{args.max_epochs}-{args.attention_type}/{args.resume_checkpoint}/checkpoints"
        files = [
            f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))
        ]
        resume_checkpoint = join(checkpoint_path, "last.ckpt")

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=-1,
        auto_select_gpus=False,
        strategy="ddp",
        logger=loggers,
        default_root_dir=args.output_path,
    )
    trainer.callbacks.append(checkpoint_last)
    trainer.callbacks.append(checkpoint_best)
    trainer.callbacks.append(early_stopping)

    # mlflow.pytorch.autolog()

    dm = MVTecADDataModule(
        args.dataset,
        args.image_type,
        args.image_size,
        args.patch_size,
        args.window_size,
        args.train_ratio,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    dm.prepare_data()

    if args.test:
        checkpoint_file = None
        checkpoint_path = f"{args.output_path}/{args.image_type}-{args.max_epochs}-{args.attention_type}/"
        dirs = [
            f for f in listdir(checkpoint_path) if not isfile(join(checkpoint_path, f))
        ]
        for dir in dirs:
            files = [
                f
                for f in listdir(join(checkpoint_path, dir, "checkpoints"))
                if isfile(join(checkpoint_path, dir, "checkpoints", f)) and "best" in f
            ]
            if len(files) > 0:
                checkpoint_file = join(checkpoint_path, dir, "checkpoints", files[0])

        print(f"Using checkpoint file {checkpoint_file}")

        train_model = InTra.load_from_checkpoint(checkpoint_file)
        dm.use_train_for_test = True
        trainer.test(train_model, dm)

        if isfile(f"{args.output_path}/{args.image_type}-{args.max_epochs}-{args.attention_type}/average_train_diff.npy"):
            average_train_diff = np.load(
                f"{args.output_path}/{args.image_type}-{args.max_epochs}-{args.attention_type}/average_train_diff.npy", allow_pickle=True)
        else:
            average_train_diff = np.sum(
                np.array(train_model.test_artifacts["amap"]), axis=0
            ) / len(train_model.test_artifacts["amap"])
            print(
                "train diff min-max", np.min(average_train_diff), np.max(average_train_diff)
            )
            np.save(f"{args.output_path}/{args.image_type}-{args.max_epochs}-{args.attention_type}/average_train_diff.npy", average_train_diff)

        del train_model
        torch.cuda.empty_cache()

        model = InTra.load_from_checkpoint(checkpoint_file)
        model.save_images = True
        model.train_diff = average_train_diff
        dm.use_train_for_test = False
        trainer.test(model, datamodule=dm)

        # cv2.imshow('image', model.test_artifacts["img"][0])
        # cv2.imshow('image_reconstruction', model.test_artifacts["reconst"][0])
        # cv2.imshow('image_mask', model.test_artifacts["gt"][0])
        # cv2.imshow('anomaly map', model.test_artifacts["amap"][0])
        # cv2.waitKey(0)

        print(model.test_artifacts["scores"])
        print(model.test_artifacts["labels"])
        print(np.array(model.test_artifacts["scores"]).shape)
        print(np.array(model.test_artifacts["labels"]).shape)
        print(np.array(model.test_artifacts["amap"]).shape)

        detection_file = f"det-result-{args.attention_type}.npy"
        segmentation_file = f"seg-result-{args.attention_type}.npy"

        detection_results = {}
        if isfile(detection_file):
            detection_results = dict(enumerate(np.load(detection_file, allow_pickle=True).flatten(), 1))[1]

        try:
            detection_metrics = compute_imagewise_retrieval_metrics(
                model.test_artifacts["scores"], model.test_artifacts["labels"]
            )
            print(detection_metrics)
            detection_results[args.image_type] = detection_metrics
        except Exception as e:
            print(e)

        segmentation_results = {}
        if isfile(segmentation_file):
            segmentation_results = dict(
                enumerate(np.load(segmentation_file, allow_pickle=True).flatten(), 1))[1]

        metrics = compute_pixelwise_retrieval_metrics(
            np.array(model.test_artifacts["amap"]),
            np.array(model.test_artifacts["gt"]),
        )
        segmentation_results[args.image_type] = metrics
        print(metrics)

        np.save(segmentation_file, np.array(segmentation_results))
        np.save(detection_file, np.array(detection_results))
    else:
        model = InTra(args)
        trainer.fit(model, dm, ckpt_path=resume_checkpoint)
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # add built-in Trainer args
    parser.add_argument("--output_path", type=str, default="./out")
    parser.add_argument("--infer", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--image_type", type=str, default="wood")
    parser.add_argument("--dataset", type=str, default="./mvtec-ad/")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.86)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=13)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--att_dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--attention_type", type=str, default="full")
    args = parser.parse_args()
    print(args)
    main(args)
