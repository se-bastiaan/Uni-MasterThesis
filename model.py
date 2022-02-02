from argparse import Namespace

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn, optim
from torch.nn import functional
from fast_transformers.builders import TransformerEncoderBuilder
from torchsummary import summary
from torchvision import utils

from loss import InTraLoss
from utils import g


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
        loss=None,
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
        self.K = patch_size
        self.L = window_size
        self.C = num_channels
        self.n_pixels = patch_size * patch_size * num_channels  # Pixels per patch
        num_patches = (image_size // patch_size) ** 2

        self.patch_to_embedding = nn.Linear(self.n_pixels, embed_dim)
        self.generator = nn.Linear(embed_dim, self.n_pixels)
        self.x_inpaint = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches * num_patches, embed_dim)
        )

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

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_avg = torch.mean(encoded, dim=1)
        generated_patch = self.generator(encoded_avg)
        return generated_patch

    def _compute_loss(self, patch_recon, patch_gt):
        mse_loss = self.mse(patch_gt, patch_recon)
        msgms_loss, msgms_map = self.msgms(patch_gt, patch_recon)
        ssim_loss, ssim_map = self.ssim(patch_gt, patch_recon)
        total_loss = mse_loss + 0.01 * msgms_loss + 0.01 * ssim_loss
        total_loss = mse_loss + msgms_loss + ssim_loss
        return total_loss, msgms_map, ssim_map

    def _preprocess_train_batch(self, image):
        # get img size
        B, C, H, W = image.size()
        # confusing notations.. we use M x N not N x M for original papers.
        M = int(H / self.K)
        N = int(W / self.K)

        # We start from 1~M*N and -1 at the pos_embedding_glb_idx -= 1 to make the range 0~M*N-1
        pos_embedding_glb_grid = torch.arange(1, N * M + 1, dtype=torch.long).reshape(
            M, N
        )

        # sampled_rs_idx : [B, 2] / sampled_rs_idx[b] = [r, s] (0 <= r <= M-L / 0 <= s <= N-L)
        sampled_rs_idx = torch.cat(
            [
                torch.randint(0, M - self.L + 1, (B, 1)),
                torch.randint(0, N - self.L + 1, (B, 1)),
            ],
            dim=1,
        )
        # pos_embedding_glb_idx : [B, L*L] (size : L*L, but 0 <= value < M*N)
        # sampled subgrid's positional embedding index : i, j(sampled_rs_idx) -> i*N + j(pos_embedding_glb_grid)
        pos_embedding_glb_idx = torch.vstack(
            [
                pos_embedding_glb_grid[r : r + self.L, s : s + self.L].unsqueeze(0)
                for r, s in sampled_rs_idx
            ]
        ).unsqueeze(dim=1)
        pos_embedding_glb_idx = pos_embedding_glb_idx.reshape(
            pos_embedding_glb_idx.size(0), -1
        )
        pos_embedding_glb_idx -= 1

        # sampled subgrid's values...
        batch_subgrid = torch.vstack(
            [
                image[
                    l,
                    :,
                    self.K * r : self.K * (r + self.L),
                    self.K * s : self.K * (s + self.L),
                ].unsqueeze(0)
                for l, (r, s) in enumerate(sampled_rs_idx)
            ]
        )

        # batch subgrid : [B, C, L*K, L*K] / corresponding positional embedding index(pos_embedding_glb_idx) : [B, L, L] (value range : 1~L**2)
        # now... convert to transformer input formats..
        # pos_embedding_glb_idx : [B, L*L]
        # pos_embedding : [1, M*N, d_model]
        # pos_embedding : [B, L*L, d_model]
        pos_embedding = torch.zeros(
            (B, self.L * self.L, self.pos_embedding.size(2))
        ).to(self.pos_embedding.device)
        for b in range(B):
            for n in range(pos_embedding_glb_idx.size(1)):
                pos_embedding[b, n, :] = self.pos_embedding[
                    :, pos_embedding_glb_idx[b, n], :
                ]

        # pos_embedding = pos_embedding[pos_embedding_glb_idx]
        # h, w : L
        batch_subgrid_flatten = rearrange(
            batch_subgrid,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.K,
            p2=self.K,
        )

        # inpaint index
        # sampled_tu_idx : [B]
        # it extracts from the flattened array with hidden dimension
        sampled_tu_idx = torch.randint(0, self.L * self.L, (B,))
        sampled_tu_idx_one_hot = functional.one_hot(sampled_tu_idx, self.L * self.L)
        sampled_tu_idx_T = sampled_tu_idx_one_hot.bool()
        sampled_tu_idx_F = torch.logical_not(sampled_tu_idx_T)

        batch_subgrid_inpaint = batch_subgrid_flatten[sampled_tu_idx_T]
        pos_embedding_inpaint = pos_embedding[sampled_tu_idx_T].unsqueeze(1)

        batch_subgrid_emb_input = batch_subgrid_flatten[sampled_tu_idx_F].reshape(
            B, self.L * self.L - 1, self.K * self.K * C
        )
        pos_embedding_emb_input = pos_embedding[sampled_tu_idx_F].reshape(
            B, self.L * self.L - 1, -1
        )

        # concat at seq dimension
        batch_subgrid_input = torch.cat(
            [
                self.x_inpaint + pos_embedding_inpaint,
                self.patch_to_embedding(batch_subgrid_emb_input)
                + pos_embedding_emb_input,
            ],
            dim=1,
        )

        return batch_subgrid_input, batch_subgrid_inpaint

    def reshape_patches(self, ground_truth, reconstruction):
        ground_truth = ground_truth.reshape(
            ground_truth.size(0), self.C, self.K, self.K
        )
        reconstruction = reconstruction.reshape(
            reconstruction.size(0), self.C, self.K, self.K
        )
        return ground_truth, reconstruction

    # def _process_one_image(self, image):
    #     image_recon, gt, loss = self._process_infer_image(image)
    #     _, msgms_map, _ = self._compute_loss(image_recon, gt)
    #
    #     return loss, image_recon, gt, msgms_map
    #
    # def _process_infer_image(self, image):
    #     patches_recon = []
    #     patches_gt = []
    #     patches_loss = []
    #     # get img size
    #     B, C, H, W = image.size()
    #     assert B == 1
    #     # confusing notations.. we use M x N not N x M for original papers.
    #     M = int(H / self.K)
    #     N = int(W / self.K)
    #
    #     for t in range(M):
    #         for u in range(N):
    #             r = g(t, self.L) - max(0, g(t, self.L) + self.L - M - 1)
    #             s = g(u, self.L) - max(0, g(u, self.L) + self.L - N - 1)
    #             subgrid_input, subgrid_inpaint = self._process_subgrid(
    #                 image,
    #                 self.x_inpaint,
    #                 self.pos_embedding,
    #                 self.patch_to_embedding,
    #                 M,
    #                 N,
    #                 r,
    #                 s,
    #                 t,
    #                 u,
    #                 self.K,
    #                 self.L,
    #             )
    #             patch_recon = self.forward(subgrid_input)
    #             patches_recon.append(patch_recon)
    #             patches_gt.append(subgrid_inpaint)
    #
    #             p_recon_r = patch_recon.reshape(
    #                 patch_recon.size(0), self.C, self.K, self.K
    #             )
    #             p_gt_r = subgrid_inpaint.reshape(
    #                 subgrid_inpaint.size(0), self.C, self.K, self.K
    #             )
    #             patches_loss.append(self._compute_loss(p_recon_r, p_gt_r)[0])
    #
    #     image_recon = self._combine_recon_patches(patches_recon, M, N, self.K)
    #     gt = self._combine_recon_patches(patches_gt, M, N, self.K)
    #     loss = torch.mean(torch.tensor(patches_loss)) / B
    #     return image_recon, gt, loss
    #
    # def _process_subgrid(self, image, M, N, r, s, t, u):
    #     # change r, s range from 1 <= r, s <= M-L+1, N-L+1
    #     #                     to 0 <= r, s <= M-L, N-L
    #     r = min(max(0, r - 1), M - self.L)
    #     s = min(max(0, s - 1), N - self.L)
    #     B, C, H, W = image.size()
    #     # subgrid -> [1, C, K*self.L, K*self.L]
    #     subgrid = image[
    #         :, :, self.K * r : self.K * (r + self.L), self.K * s : self.K * (s + self.L)
    #     ]
    #     # subgrid_flatten : [1, self.L*self.L, self.K*self.K*C]
    #     subgrid_flatten = rearrange(
    #         subgrid, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.K, p2=self.K
    #     )
    #
    #     # pos_embedding_glb_idx : [1, L*L]
    #     pos_embedding_glb_grid = torch.arange(1, M * N + 1, dtype=torch.long).reshape(
    #         M, N
    #     )
    #     pos_embedding_glb_idx = pos_embedding_glb_grid[
    #         r : r + self.L, s : s + self.L
    #     ].unsqueeze(0)
    #     pos_embedding_glb_idx = pos_embedding_glb_idx.reshape(
    #         pos_embedding_glb_idx.size(0), -1
    #     )
    #     pos_embedding_glb_idx -= 1
    #
    #     # pos_embedding_grid : [1, self.L*self.L, d_model]
    #     pos_embedding = torch.zeros(1, self.L * self.L, self.pos_embedding.size(2)).to(
    #         self.pos_embedding.device
    #     )
    #     for n in range(pos_embedding_glb_idx.size(1)):
    #         pos_embedding[:, n, :] = self.pos_embedding[
    #             :, pos_embedding_glb_idx[:, n], :
    #         ]
    #
    #     # r, s, t, u ... M x N
    #     # t, u : 0 <= t <= M / 0 <= u <= n
    #
    #     # tu_1d_idx : 0 <= val < M*N
    #     # but it should be shape in L*L
    #     tu_1d_idx = torch.tensor([(t - r) * self.L + (u - s)], dtype=torch.long)
    #     tu_one_hot = functional.one_hot(tu_1d_idx, self.L * self.L)
    #     tu_idx_T = tu_one_hot.bool()
    #     tu_idx_F = torch.logical_not(tu_idx_T)
    #
    #     subgrid_inpaint = subgrid_flatten[tu_idx_T]
    #     pos_embedding_inpaint = pos_embedding[tu_idx_T].unsqueeze(1)
    #
    #     subgrid_emb_input = subgrid_flatten[tu_idx_F].reshape(
    #         B, self.L * self.L - 1, self.K * self.K * C
    #     )
    #     pos_embedding_emb_input = pos_embedding[tu_idx_F].reshape(
    #         B, self.L * self.L - 1, -1
    #     )
    #
    #     subgrid_input = torch.cat(
    #         [
    #             self.x_inpaint + pos_embedding_inpaint,
    #             self.patch_to_embedding(subgrid_emb_input) + pos_embedding_emb_input,
    #         ],
    #         dim=1,
    #     )
    #     return subgrid_input, subgrid_inpaint
    #
    # def _combine_recon_patches(self, patch_list, M, N, K):
    #     # patch_list : list of M*N [1, K*K*C] tensor
    #     # patches_concat : [M*N, 1, K*K*C]
    #     patch_list = [x.unsqueeze(0) for x in patch_list]
    #     patches_concat = torch.cat(patch_list, dim=0)
    #     # patches_concat : [1, M*N, K*K*C]
    #     patches_concat = patches_concat.permute(1, 0, 2)
    #     # recon_image : [1, C, H, W]
    #     recon_image = rearrange(
    #         patches_concat,
    #         "b (h w) (p1 p2 c) -> b c (h p1) (w p2) ",
    #         h=M,
    #         w=N,
    #         p1=K,
    #         p2=K,
    #     )
    #     return recon_image

    def _combine_recon_patches(self, patch_list, M, N):
        # patch_list : list of M*N [1, patch_size*patch_size*C] tensor
        # patches_concat : [M*N, 1, patch_size*patch_size*C]
        patch_list = [x.unsqueeze(0) for x in patch_list]
        patches_concat = torch.cat(patch_list, dim=0)
        # patches_concat : [1, M*N, patch_size*patch_size*C]
        patches_concat = patches_concat.permute(1, 0, 2)
        # recon_image : [1, C, H, W]
        recon_image = rearrange(
            patches_concat,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2) ",
            h=M,
            w=N,
            p1=self.K,
            p2=self.K,
        )
        return recon_image


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

        summary(self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), input_size=(512, 512))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, Y_pred, Y, mode="train"):
        l2_loss, ssim_loss, ssim_map, msgms_loss, msgms_map, total_loss = self.loss(
            Y_pred, Y
        )

        self.log(f"{mode}_loss", total_loss)
        self.log(f"{mode}_L2_loss", l2_loss)
        self.log(f"{mode}_GMS_loss", msgms_loss)
        self.log(f"{mode}_SSIM_loss", ssim_loss)
        return total_loss, msgms_map

    def _step(self, batch, mode):
        x, ground_truth = self.model._preprocess_train_batch(
            batch[0]
        )  # 0 is the index of the image
        reconstruction = self.model(x)
        ground_truth, reconstruction = self.model.reshape_patches(
            ground_truth, reconstruction
        )
        loss, msgms_map = self._calculate_loss(reconstruction, ground_truth, mode=mode)
        return ground_truth, reconstruction, loss, msgms_map

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
                torch.cat([Y[:10], Y_pred[:10], msgms_img[:10]], dim=0),
                nrow=10,
                normalize=False,
            )
            self.logger[0].experiment.add_image(
                "generated_images", grid, self.current_epoch
            )
        return loss

    def test_step(self, batch, batch_idx):
        image_recon, gt, loss = self._process_infer_image(batch[0])
        _, msgms_map, _ = self._compute_loss(image_recon, gt)
        return loss
