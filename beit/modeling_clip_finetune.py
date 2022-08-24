# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    # def __init__(self):
    #     super(nn.LayerNorm, self).__init__(eps=1e-6)

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path = 0.0, init_values=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((d_model)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((d_model)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # attn_x, attn_weight = self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask)
        attn_x, attn_weight = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)
        return attn_x

    def forward(self, x: torch.Tensor):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attention(self.ln_1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln_2(x)))
        else:
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate = 0.0, init_values = None):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask, dpr[_], init_values=init_values) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, num_classes: int, drop_path_rate=0.0, init_values=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.width = width
        self.scale = width ** -0.5
        self.class_embedding = nn.Parameter(self.scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(self.scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate, init_values=init_values)

        self.ln_post = LayerNorm(width)
        self.head = nn.Parameter(self.scale * torch.randn(width, num_classes))
        # trunc_normal_(self.head, std=.02)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.use_rel_pos_bias = False
    def get_num_layers(self):
        return self.transformer.layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Parameter(self.scale * torch.randn(self.width, num_classes)) if num_classes > 0 else None

    def forward_featuremap(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_featuremap(x)
        x = self.ln_post(x[:, 0, :])

        if self.head is not None:
            x = x @ self.head

        return x


@register_model
def clip_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        input_resolution=224, patch_size=16, width=768, layers=12, heads=12, 
        **kwargs)
    model.default_cfg = _cfg()
    return model
