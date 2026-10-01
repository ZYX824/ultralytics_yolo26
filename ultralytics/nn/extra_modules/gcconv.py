"""Multi-branch GCConv with equivalent single-convolution inference."""

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_weights

from ..modules.conv import Conv

__all__ = ("GCConv",)


class GCConv(nn.Module):
    """Sum two 3x3-to-1x1 paths, a 1x1-to-1x1 path, and an optional BN residual before SiLU."""

    def __init__(self, c1, c2, k=3, s=1, p=1, deploy=False):
        """Initialize GCConv with the branch structure from the reference YOLOv26 implementation.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size, which must be 3.
            s (int): Convolution stride.
            p (int): Padding, which must be 1.
            deploy (bool): Build only the equivalent convolution for loading fused weights.
        """
        super().__init__()
        assert k == 3 and p == 1, "GCConv requires a 3x3 kernel with padding 1"
        self.act = nn.SiLU()
        if deploy:
            self.reparam_3x3 = nn.Conv2d(c1, c2, k, s, p, bias=True)
        else:
            self.branches = nn.ModuleList(
                nn.Sequential(Conv(c1, c2, kernel, s, kernel // 2, act=False), Conv(c2, c2, 1, act=False))
                for kernel in (3, 3, 1)
            )
            self.path_residual = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

    def forward(self, x):
        """Apply the training branches or the equivalent inference convolution.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            (torch.Tensor): Activated output feature map.
        """
        if hasattr(self, "reparam_3x3"):
            return self.act(self.reparam_3x3(x))
        residual = self.path_residual(x) if self.path_residual is not None else 0
        return self.act(sum(branch(x) for branch in self.branches) + residual)

    def get_equivalent_kernel_bias(self):
        """Compute equivalent inference weights without modifying the training branches.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Equivalent 3x3 kernel and bias.
        """
        kernels, biases = [], []
        for branch in self.branches:
            fused = [
                fuse_conv_bn_weights(
                    layer.conv.weight,
                    layer.conv.bias,
                    layer.bn.running_mean,
                    layer.bn.running_var,
                    layer.bn.eps,
                    layer.bn.weight,
                    layer.bn.bias,
                )
                for layer in branch
            ]
            (kernel1, bias1), (kernel2, bias2) = fused
            mixing = kernel2[:, :, 0, 0]
            kernel = torch.einsum("oi,icjk->ocjk", mixing, kernel1)
            kernels.append(torch.nn.functional.pad(kernel, (1, 1, 1, 1)) if kernel.shape[-1] == 1 else kernel)
            biases.append(bias2 + mixing @ bias1)

        kernel, bias = sum(kernels), sum(biases)
        if self.path_residual is not None:
            bn = self.path_residual
            gain = bn.weight / (bn.running_var + bn.eps).sqrt()
            identity = torch.zeros_like(kernel)
            indices = torch.arange(kernel.shape[0], device=kernel.device)
            identity[indices, indices, 1, 1] = gain
            kernel = kernel + identity
            bias = bias + bn.bias - bn.running_mean * gain
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self):
        """Replace the branches with their equivalent convolution using evaluation-mode BN statistics."""
        if hasattr(self, "reparam_3x3"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = self.branches[0][0].conv
        self.reparam_3x3 = (
            nn.Conv2d(conv.in_channels, conv.out_channels, 3, conv.stride, 1, bias=True)
            .to(device=kernel.device, dtype=kernel.dtype)
            .requires_grad_(False)
        )
        self.reparam_3x3.weight.copy_(kernel)
        self.reparam_3x3.bias.copy_(bias)
        del self.branches, self.path_residual
