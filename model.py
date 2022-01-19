from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch import nn, optim
from fast_transformers.builders import TransformerEncoderBuilder
from torchvision import utils

from loss import InTraLoss


class InTraModel(nn.Module):
    def __init__(
        self,
        attention_type: str,
        image_size: int,
        patch_size: int = 16,
        window_size: int = 7,
        num_layers: int = 13,
        num_heads: int = 8,
        embed_dim: int = 512,
        num_channels: int = 3,
        dropout: float = 0,
        att_dropout=None,
    ):
        """
        grid_size_max:
        patch_size: This is the desired side length of a square patch. Called K in the paper. (following ViT, K=16)
        window_size: This the side length of a subgrid in the full grid that is used as context window. Called L in the paper.
        num_channels: Number of
        num_layers: Number of layers in the encoder
        num_heads: Number of attention heads
        embed_dim: Embedding size
        dropout: Encoder dropout
        att_dropout: Dropout of the attention
        attention_type: Type of the attention used
        """
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image size should be multiple of patch size."
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_channels = num_channels
        num_patches = (image_size // patch_size) ** 2

        super().__init__()

        # Project (K*K*C) dim patch into D dim embedding
        self.linear_projection = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        # Project D dim embedding into (K*K*C) dim patch.
        self.affine_projection = nn.Linear(embed_dim, num_channels * (patch_size ** 2))

        # Learnable parameters for positional embedding.
        # There will be total M*N embedding,
        # but in forward pass, we will dynamically select (48+1) embedding
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, embed_dim))
        # Learnable parameter which will be inserted on behalf of inpaint-patch.
        self.x_inpaint = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=num_layers,
            n_heads=num_heads,
            feed_forward_dimensions=embed_dim * 4,
            attention_type=attention_type,
            query_dimensions=embed_dim // num_heads,
            value_dimensions=embed_dim // num_heads,
            dropout=dropout,
            attention_dropout=att_dropout,
            final_normalization=False,
            activation="gelu",
        )

        self.encoder = builder.get()

    def forward(self, x_nbr, patch_positions, inpaint_position):
        """
        Let's say, if window size L=7, patch size K=16, channel=3,
                   (L*L)-1 --> 48;  (C*K*K) --> 768
        Args:
            x_nbr (tensor):            [B, 48, 768] dim float
            patch_positions (tensor):  [B, 48]      dim int
            inpaint_position (tensor): [B, 1]       dim int
        """
        B = x_nbr.shape[0]  # batch_size
        # linear projection from (C*K*K) to D dimensional embedding
        x_nbr = self.linear_projection(x_nbr)
        # copy x_inpaint learnable parameter for all samples in the batch
        x_inp = self.x_inpaint.repeat(B, 1, 1)

        # dynamically choose positional embedding from all(N*M).
        # i.e.: (B, 48, 512)
        neighbors_pos_embedding = self.pos_embedding[patch_positions, :]
        # i.e.: (B,  1, 512)
        inpaint_pos_embedding = self.pos_embedding[inpaint_position, :]

        # Add positional-embedding with patch projection-embedding
        x_nbr = x_nbr + neighbors_pos_embedding
        x_inp = x_inp + inpaint_pos_embedding

        # Create (L.L) * D dimensional embedding for transformer block
        x = torch.cat([x_inp, x_nbr], dim=1)

        x = self.encoder(x)

        # [B*(L*L)*D] dim to [B, 1, D] dim by averaging.
        x = torch.mean(x, dim=1)

        # Project back to patch dimension(flatten state)
        # [B, 1, D] --> [B, (K*K*C)]
        x = self.affine_projection(x)
        x = x.reshape(B, self.num_channels, self.patch_size, self.patch_size)
        return x


class InTra(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters(args)
        self.loss = InTraLoss()
        self.last_epoch = self.current_epoch
        self.model = InTraModel(
            image_size=args.image_size,
            patch_size=args.patch_size,
            window_size=args.window_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            num_channels=args.num_channels,
            dropout=args.dropout,
            att_dropout=args.att_dropout,
            attention_type=args.attention_type,
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, Y_pred, Y, mode="train"):
        l2_loss, ssim_loss, ssim_map, msgms_loss, msgms_map, total_loss = self.loss(Y_pred, Y)

        self.log(f"{mode}_loss", total_loss)
        self.log(f"{mode}_L2_loss", l2_loss)
        self.log(f"{mode}_GMS_loss", msgms_loss)
        self.log(f"{mode}_SSIM_loss", ssim_loss)
        return total_loss, msgms_map

    def _step(self, batch, mode):
        patches_info, labels = batch
        X = patches_info["neighbor_patchs"]
        Y = patches_info["inpaint_patch"]
        X_positions = patches_info["neighbor_positions"]
        Y_position = patches_info["inpaint_position"]

        Y_pred = self.model(X, X_positions, Y_position)
        loss, msgms_map = self._calculate_loss(Y_pred, Y, mode=mode)
        return Y, Y_pred, loss, msgms_map

    def training_step(self, batch, batch_idx):
        _, _, loss, _ = self._step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        Y, Y_pred, loss, msgms_map = self._step(batch, mode="val")
        msgms_img = msgms_map.repeat(1, 3, 1, 1)

        # save 10 images for logging
        if self.current_epoch - self.last_epoch > 0:
            self.last_epoch += 1
            grid = utils.make_grid(
                torch.cat([Y[:10], Y_pred[:10], msgms_img[:10]], dim=0), nrow=10, normalize=False
            )
            self.logger[0].experiment.add_image(
                "generated_images", grid, self.current_epoch
            )
        return loss

    def test_step(self, batch, batch_idx):
        _, _, loss, _ = self._step(batch, mode="test")
        return loss
