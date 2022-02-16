from argparse import ArgumentParser
from os import listdir
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
from model import InTra
from utils import tensor2nparr, compute_auroc

IMAGE_SIZE = {
    'carpet': 512,
    'grid': 256,
    'leather': 512,
    'tile': 512,
    'wood': 512,
    'bottle': 256,
    'cable': 256,
    'capsule': 320,
    'hazelnut': 256,
    'metal_nut': 256,
    'pill': 512,
    'screw': 320,
    'toothbrush': 256,
    'transistor': 256,
    'zipper': 512
}

def main(args):
    args.image_size = args.image_size if args.image_size else IMAGE_SIZE[args.image_type]

    pl.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Sees device " + str(device))

    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/", name=f"{args.image_type}-{args.max_epochs}-{args.attention_type}"
    )
    loggers = [tb_logger]

    checkpoint_best = ModelCheckpoint(
        filename='best-{epoch}-{step}-{val_loss:.5f}',
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    checkpoint_last = ModelCheckpoint(
        filename='last-{epoch}-{step}',
        save_last=True,
        monitor="val_loss",
        mode="min",
    )

    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        checkpoint_path = args.checkpoint_path + f"/{args.image_type}-{args.max_epochs}-{args.attention_type}/{args.resume_checkpoint}/checkpoints"
        files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
        resume_checkpoint = join(checkpoint_path, files[-1])

    trainer = pl.Trainer.from_argparse_args(
        args, gpus=1, logger=loggers, default_root_dir=args.checkpoint_path, resume_from_checkpoint=resume_checkpoint
    )
    trainer.callbacks.append(checkpoint_last)
    trainer.callbacks.append(checkpoint_best)
    trainer.callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=50)
    )

    mlflow.pytorch.autolog()

    dm = MVTecADDataModule(
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

    if args.load_checkpoint is not None:
        checkpoint_path = args.checkpoint_path + f"/{args.image_type}-{args.max_epochs}-{args.attention_type}/{args.load_checkpoint}/checkpoints"
        files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
        checkpoint_file = join(checkpoint_path, files[0])
        if args.infer:
            mod = InTra.load_from_checkpoint(checkpoint_file)
            model = mod.model
            model.eval()

            test_loss = 0
            amaps = []
            gt = []

            with torch.no_grad():
                with tqdm(dm.test_dataloader(), unit="batch") as loader:
                    for data, label in loader:
                        data = data.to(device)
                        loss, image_recon, image_reassembled, msgms_map = model._process_one_image(data, mod._calculate_loss)
                        test_loss += loss.detach().cpu().numpy()

                        gt.append(tensor2nparr(data))
                        amaps.append(tensor2nparr(msgms_map))

                        data_arr = tensor2nparr(data)
                        image_recon_arr = tensor2nparr(image_recon)
                        image_reassembled_arr = tensor2nparr(image_reassembled)
                        msgms_map_arr = tensor2nparr(msgms_map)

                        cv2.imshow('image', data_arr[0])
                        cv2.imshow('image_recon_arr', image_recon_arr[0])
                        cv2.imshow('image_reassembled_arr', image_reassembled_arr[0])
                        cv2.imshow('msgms_map_arr', msgms_map_arr[0])
                        cv2.imshow('heatmap', cv2.applyColorMap(msgms_map_arr[0], cv2.COLORMAP_JET))
                        cv2.waitKey(0)
                print(test_loss)
                print(compute_auroc(0, np.array(amaps), np.array(gt)))
        else:
            model = InTra.load_from_checkpoint(checkpoint_file)
            result = trainer.test(model, datamodule=dm)
            print(result)

            ep_amap = np.array(model.test_artifacts["amap"])
            ep_amap = (ep_amap - ep_amap.min()) / (ep_amap.max() - ep_amap.min())
            model.test_artifacts["amap"] = list(ep_amap)

            cv2.imshow('image', model.test_artifacts["img"][0])
            cv2.imshow('image_reconstruction', model.test_artifacts["reconst"][0])
            cv2.imshow('image_mask', model.test_artifacts["gt"][0])
            cv2.imshow('anomaly map', model.test_artifacts["amap"][0])
            cv2.imshow('heatmap', cv2.applyColorMap(model.test_artifacts["amap"][0], cv2.COLORMAP_JET))
            cv2.waitKey(0)

            auroc = compute_auroc(
                0,
                np.array(model.test_artifacts["amap"]),
                np.array(model.test_artifacts["gt"])
            )
            print(auroc)

    else:
        model = InTra(args)
        trainer.fit(model, dm)
        trainer.test(ckpt_path="best", datamodule=dm)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # add built-in Trainer args
    parser.add_argument("--checkpoint_path", type=str, default='./ckpt')
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--infer", action='store_true', default=False)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--image_type", type=str, default="wood")
    parser.add_argument("--dataset", type=str, default="./mvted-ad/")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.86)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
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
