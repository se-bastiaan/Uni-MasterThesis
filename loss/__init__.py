from typing import Tuple

from torch.nn.modules import Module
from torch import nn

from torch import Tensor

from .msgms import MSGMSLoss
from .ssim import SSIMLoss


class InTraLoss(Module):
    def __init__(self, alpha: int = 0.01, beta: int = 0.01) -> None:
        super().__init__()
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.ssim_loss = SSIMLoss(kernel_size=11, sigma=1.5)
        self.gmsd_loss = MSGMSLoss(num_scales=3, in_channels=3)
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, input: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        l2_loss = self.l2_loss(input, target)
        ssim_loss, ssim_map = self.ssim_loss(input, target)
        msgms_loss, msgms_map = self.gmsd_loss(input, target)

        total_loss = l2_loss + (self.alpha * ssim_loss) + (self.beta * msgms_loss)

        return l2_loss, ssim_loss, ssim_map, msgms_loss, msgms_map, total_loss

