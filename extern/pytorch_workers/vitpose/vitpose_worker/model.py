from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ntuple(n: int):
    def parse(value: Any):
        if isinstance(value, Iterable) and not isinstance(value, str):
            return tuple(value)
        return tuple(repeat(value, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn(
                "mean is more than 2 std from [a, b] in trunc_normal_",
                stacklevel=2,
            )
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = int(attn_head_dim or dim // num_heads)
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, -1)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Any = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] | int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        ratio: int = 1,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        stride = (patch_size[0] // ratio, patch_size[1] // ratio)
        padding = 4 + 2 * (ratio // 2 - 1)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        height, width = int(x.shape[2]), int(x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, (height, width)


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] | int = (256, 192),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        ratio: int = 1,
        last_norm: bool = True,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ratio=ratio,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[index],
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        patch_height, patch_width = self.patch_embed.patch_shape
        pos = self.pos_embed[:, 1:]
        if int(pos.shape[1]) == height * width:
            return pos
        pos = pos.reshape(1, patch_height, patch_width, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, height * width, -1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch = int(x.shape[0])
        x, (height, width) = self.patch_embed(x)
        x = x + self._position_embedding(height, width) + self.pos_embed[:, :1]
        for block in self.blocks:
            x = block(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).reshape(batch, -1, height, width).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class HeatmapHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: int = 4,
        final_conv_kernel: int = 3,
        num_deconv_layers: int = 0,
        num_deconv_filters: tuple[int, ...] = (),
        num_deconv_kernels: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.upsample = int(upsample)
        if num_deconv_layers:
            self.deconv_layers = self._make_deconv_layer(
                in_channels,
                num_deconv_layers,
                tuple(num_deconv_filters),
                tuple(num_deconv_kernels),
            )
            head_channels = int(num_deconv_filters[-1])
        else:
            self.deconv_layers = nn.Identity()
            head_channels = int(in_channels)
        if final_conv_kernel == 0:
            self.final_layer = nn.Identity()
        else:
            padding = 1 if final_conv_kernel == 3 else 0
            self.final_layer = nn.Conv2d(
                head_channels,
                out_channels,
                kernel_size=final_conv_kernel,
                stride=1,
                padding=padding,
            )
        self.apply(self._init_weights)

    @staticmethod
    def _deconv_cfg(kernel_size: int) -> tuple[int, int, int]:
        if kernel_size == 4:
            return 4, 1, 0
        if kernel_size == 3:
            return 3, 1, 1
        if kernel_size == 2:
            return 2, 0, 0
        raise ValueError(f"unsupported deconv kernel size: {kernel_size}")

    def _make_deconv_layer(
        self,
        in_channels: int,
        num_layers: int,
        num_filters: tuple[int, ...],
        num_kernels: tuple[int, ...],
    ) -> nn.Sequential:
        if num_layers != len(num_filters) or num_layers != len(num_kernels):
            raise ValueError("deconv layer/filter/kernel counts must match")
        layers: list[nn.Module] = []
        channels = in_channels
        for index in range(num_layers):
            kernel, padding, output_padding = self._deconv_cfg(num_kernels[index])
            planes = int(num_filters[index])
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight, std=0.001)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.001)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample > 0:
            x = F.interpolate(
                F.relu(x),
                scale_factor=self.upsample,
                mode="bilinear",
                align_corners=False,
            )
        x = self.deconv_layers(x)
        return self.final_layer(x)


@dataclass(frozen=True)
class ViTPoseModelConfig:
    image_size: tuple[int, int]
    heatmap_size: tuple[int, int]
    nkeypoints: int
    max_objects: int
    variant: str
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    drop_path_rate: float
    upsample: int
    final_conv_kernel: int
    num_deconv_layers: int
    num_deconv_filters: tuple[int, ...]
    num_deconv_kernels: tuple[int, ...]


VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    "tiny": {"embed_dim": 64, "depth": 2, "num_heads": 4, "drop_path_rate": 0.05},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 12, "drop_path_rate": 0.2},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "drop_path_rate": 0.3},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "drop_path_rate": 0.5},
    "huge": {"embed_dim": 1280, "depth": 32, "num_heads": 16, "drop_path_rate": 0.55},
}


class ViTPoseSlots(nn.Module):
    def __init__(self, config: ViTPoseModelConfig) -> None:
        super().__init__()
        self.nkeypoints = int(config.nkeypoints)
        self.max_objects = int(config.max_objects)
        self.image_size = tuple(config.image_size)
        self.heatmap_size = tuple(config.heatmap_size)
        self.backbone = ViT(
            img_size=(config.image_size[1], config.image_size[0]),
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
        )
        self.keypoint_head = HeatmapHead(
            in_channels=config.embed_dim,
            out_channels=config.max_objects * config.nkeypoints,
            upsample=config.upsample,
            final_conv_kernel=config.final_conv_kernel,
            num_deconv_layers=config.num_deconv_layers,
            num_deconv_filters=config.num_deconv_filters,
            num_deconv_kernels=config.num_deconv_kernels,
        )
        self.objectness_head = nn.Linear(config.embed_dim, config.max_objects)
        nn.init.normal_(self.objectness_head.weight, std=0.001)
        nn.init.constant_(self.objectness_head.bias, -2.0)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        heatmaps = self.keypoint_head(features)
        batch, _channels, height, width = heatmaps.shape
        heatmaps = heatmaps.reshape(
            batch,
            self.max_objects,
            self.nkeypoints,
            height,
            width,
        )
        pooled = features.mean(dim=(2, 3))
        objectness = self.objectness_head(pooled)
        return {"heatmaps": heatmaps, "objectness": objectness}
