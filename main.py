from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import mlflow.pytorch
from pytorch_lightning.callbacks import EarlyStopping

from dataset import MVTecADDataModule
from model import InTra


def main(args):
    pl.seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Sees device " + str(device))

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    loggers = [tb_logger]

    trainer = pl.Trainer.from_argparse_args(args, gpus=1, logger=loggers)
    trainer.callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=500)
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
    model = InTra(args)

    trainer.fit(model, dm)

    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # add built-in Trainer args
    parser.add_argument("--image_type", type=str, default="wood")
    parser.add_argument("--dataset", type=str, default="./mvted-ad/")
    parser.add_argument("--image_size", type=int, default=512)
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
