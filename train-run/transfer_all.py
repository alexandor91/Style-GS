#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
from copy import deepcopy
import subprocess

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

import cv2
import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf

# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l2_loss, l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render_sh, render_depth_sh, network_gui


import sys
from scene import SceneSH, GaussianSHModel

# from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.general_utils import get_expon_lr_func

from utils.loss_utils import style_loss_adain, gram_matrix
from utils.extractor_utils import VGG19Extractor, DINOV2SExtractor
from functools import partial
import collections.abc
from itertools import repeat
from torch import nn

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net="vgg").to("cuda")

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = [".git", ".gitignore"]
    ignorePatterns = set()
    ROOT = "."
    with open(os.path.join(ROOT, ".gitignore")) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith("#"):
                if line.endswith("\n"):
                    line = line[:-1]
                if line.endswith("/"):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print("Backup Finished!")


##### Attention utils ###
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class RoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, D, 2).float().to(device) / D)
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert (
            tokens.size(3) % 2 == 0
        ), "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        cos, sin = self.get_cos_sin(
            D, int(positions.max()) + 1, tokens.device, tokens.dtype
        )
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class CrossAttention(nn.Module):
    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    # def forward(self, query, key, value, qpos, kpos):
    #     B, Nq, C = query.shape
    #     Nk = key.shape[1]
    #     Nv = value.shape[1]

    #     q = (
    #         self.projq(query)
    #         .reshape(B, Nq, self.num_heads, C // self.num_heads)
    #         .permute(0, 2, 1, 3)
    #     )
    #     k = (
    #         self.projk(key)
    #         .reshape(B, Nk, self.num_heads, C // self.num_heads)
    #         .permute(0, 2, 1, 3)
    #     )
    #     v = (
    #         self.projv(value)
    #         .reshape(B, Nv, self.num_heads, C // self.num_heads)
    #         .permute(0, 2, 1, 3)
    #     )

    #     if self.rope is not None:
    #         q = self.rope(q, qpos)
    #         k = self.rope(k, kpos)

    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = (
            self.projq(query)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.projk(key)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.projv(value)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # if self.rope is not None:
        #     q = self.rope(q, qpos)
        #     k = self.rope(k, kpos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    # def forward(self, x, xpos):
    #     B, N, C = x.shape

    #     qkv = (
    #         self.qkv(x)
    #         .reshape(B, N, 3, self.num_heads, C // self.num_heads)
    #         .transpose(1, 3)
    #     )
    #     q, k, v = [qkv[:, :, i] for i in range(3)]
    #     # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

    #     if self.rope is not None:
    #         q = self.rope(q, xpos)
    #         k = self.rope(k, xpos)

    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, x):
        """disable rope"""
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        # if self.rope is not None:
        #     q = self.rope(q, xpos)
        #     k = self.rope(k, xpos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # disable self attention
        # self.attn = Attention(
        #     dim,
        #     rope=rope,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )
        self.cross_attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    # def forward(self, x, y, xpos, ypos):
    #     # croco net deblock backbone
    #     x = x + self.drop_path(self.attn(self.norm1(x), xpos))
    #     y_ = self.norm_y(y)
    #     x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
    #     x = x + self.drop_path(self.mlp(self.norm3(x)))
    #     return x, y
    def forward(self, x, y):
        # croco net deblock backbone
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


# def block_matmul(A, B, block_size):
#     m, k = A.shape
#     _, n = B.shape
#     C = torch.zeros(m, n, dtype=A.dtype, device=A.device)

#     for i in range(0, m, block_size):
#         for j in range(0, n, block_size):
#             for l in range(0, k, block_size):
#                 A_block = A[i : min(i + block_size, m), l : min(l + block_size, k)]
#                 B_block = B[l : min(l + block_size, k), j : min(j + block_size, n)]
#                 C_block = C[i : min(i + block_size, m), j : min(j + block_size, n)]

#                 C[
#                     i : min(i + block_size, m), j : min(j + block_size, n)
#                 ] = C_block + torch.matmul(A_block, B_block)

#     return C


class StyleTransformer:
    """
    用于管理style transfer的所有功能
    参数更新 & Loss管理
    DGSO
    基于 sh 实现
    """

    def __init__(
        self,
        model=True,
    ):
        # follow https://github.com/xunhuang1995/AdaIN-style/blob/6d8ac287f35010faf8feece4047bfe86c8c007d0/train.lua#L39 setting
        # using 2 7 16 25 as style
        # using 25 as content
        self.vgg_extractor = VGG19Extractor(out_indices=[2, 7, 16, 25])
        self.dino_extractor = DINOV2SExtractor(out_indices=[2, 5, 8, 11])
        self.image_cache = {}

        ### model only for sh prediction
        # 4 layers
        self.net = nn.ModuleDict(
            {
                "proj": nn.Linear(48 * 10, 384),
                "block_0": DecoderBlock(dim=384, num_heads=6),
                "block_1": DecoderBlock(dim=384, num_heads=6),
                "block_2": DecoderBlock(dim=384, num_heads=6),
                "block_3": DecoderBlock(dim=384, num_heads=6),
                "head": nn.Linear(384, 3 * 10),
            }
        )
        self.net.cuda()
        self.net.train()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

    # def training_setup(
    #     self,
    #     opt: OptimizationParams,
    #     gaussians: GaussianModel,
    # ):
    #     l = []  # parameters
    #     gaussians.mlp_color.train()
    #     l.append(
    #         {
    #             "params": gaussians.mlp_color.parameters(),
    #             "lr": opt.style_mlp_color_lr_init,
    #             "name": "style_mlp_color",
    #         }  # appearance later
    #     )
    #     # dgso no need
    #     # self.schedulers["style_mlp_color"] = get_expon_lr_func(
    #     #     lr_init=opt.style_mlp_color_lr_init,
    #     #     lr_final=opt.style_mlp_color_lr_final,
    #     #     lr_delay_mult=opt.style_mlp_color_lr_delay_mult,
    #     #     max_steps=opt.style_mlp_color_lr_max_steps,
    #     # )
    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    # def update_learning_rate(self, iteration):
    #     return
    #     # for param_group in self.optimizer.param_groups:
    #     #     lr = self.schedulers[param_group["name"]](iteration)
    #     #     param_group["lr"] = lr
    #     # for debugging
    # print(param_group["name"], " update lr: ", lr)
    def save_model(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(savedir, "net.pth"))

    def load_model(self, savedir):
        self.net.load_state_dict(torch.load(os.path.join(savedir, "net.pth")))
        self.net.eval()

    @torch.no_grad()
    def update_image_cache(self, style_img):
        """缓存style image的特征
        这里进行color matching，保存图像的feature和VGG特征，不要保存新图的原图
        包含原始图像 图像统计数据 VGG特征 DINO特征；
        这里gram预先计算一下，然后仿射变换可能需要估计
        - 颜色仿射变换可能无法估计, size不一样
        - VGG layer's feature
        - VGG layer's grammer matrix
        """
        img = cv2.imread(style_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()

        cache = {}
        # calc statistic
        source_img = img.squeeze(0)
        source_h, source_w = source_img.shape[1], source_img.shape[2]
        source_mean = torch.mean(source_img, dim=[1, 2])
        source_vec = source_img.view(3, -1) - source_mean.view(3, -1)
        source_cov = source_vec.matmul(source_vec.T) / (
            source_h * source_w
        ) + 1e-5 * torch.eye(3, device=source_vec.device)
        source_chol = torch.linalg.cholesky(source_cov)

        cache["img_mean"] = source_mean
        cache["img_chol"] = source_chol

        # rgb
        # cache["rgb_mean"] = torch.mean(img, dim=[0, 2, 3])
        # cache["rgb_std"] = torch.std(img, dim=[0, 2, 3])
        # vgg
        cache.update(self.get_vgg_feat(img))
        # dino
        cache.update(self.get_dino_feat(img))

        self.image_cache[style_img] = cache

    def color_matching(self, style_img, gt_img):
        # target to be matched

        target_img = gt_img
        target_h, target_w = target_img.shape[1], target_img.shape[2]
        target_mean = torch.mean(target_img, dim=[1, 2])
        target_vec = target_img.view(3, -1) - target_mean.view(3, -1)
        target_cov = target_vec.matmul(target_vec.T) / (
            target_h * target_w
        ) + 1e-5 * torch.eye(3, device=target_vec.device)
        target_chol = torch.linalg.cholesky(target_cov)

        # source : the style
        source_mean = self.image_cache[style_img]["img_mean"]
        source_chol = self.image_cache[style_img]["img_chol"]

        target_vec_trans = source_chol.matmul(torch.linalg.inv(target_chol)).matmul(
            target_vec
        )
        target_img_trans = target_vec_trans.reshape(
            [3, target_h, target_w]
        ) + source_mean.view(3, 1, 1)

        return target_img_trans

    def color_matching_preserve(self, style_img, gt_img):
        # target to be matched
        target_img = gt_img
        target_img_g = (
            0.299 * target_img[0] + 0.587 * target_img[1] + 0.114 * target_img[2]
        )
        # target_mask_g = torch.logical_and(target_img_g > 0.1, target_img_g < 0.9)

        # target_mean = torch.mean(target_img[:, target_mask_g], dim=-1, keepdim=True)
        # target_vec = target_img[:, target_mask_g] - target_mean
        # target_cov = target_vec.matmul(target_vec.T) / (
        #     target_vec.shape[-1]
        # ) + 1e-5 * torch.eye(3, device=target_vec.device)
        # target_chol = torch.linalg.cholesky(target_cov)
        target_mask_all = target_img_g > -1.0
        target_mean = torch.mean(target_img[:, target_mask_all], dim=-1, keepdim=True)
        target_vec = target_img[:, target_mask_all] - target_mean
        target_cov = target_vec.matmul(target_vec.T) / (
            target_vec.shape[-1]
        ) + 1e-5 * torch.eye(3, device=target_vec.device)
        target_chol = torch.linalg.cholesky(target_cov)

        source_mean = self.image_cache[style_img]["img_mean"]
        source_chol = self.image_cache[style_img]["img_chol"]

        target_vec_trans = source_chol.matmul(torch.linalg.inv(target_chol)).matmul(
            target_vec
        )
        target_vec_trans = target_vec_trans + source_mean.view(3, 1)
        target_img_trans = target_img.clone()
        target_img_trans[:, target_mask_all] = target_vec_trans
        
        target_img_trans_g = (
            0.299 * target_img_trans[0]
            + 0.587 * target_img_trans[1]
            + 0.114 * target_img_trans[2]
        )

        g_scale = target_img_g / (target_img_trans_g+1e-5)
        target_img_trans=target_img_trans*g_scale[None,...]

        # torchvision.utils.save_image(target_img_trans, "./trans.png")
        return target_img_trans

    def color_matching_naiive(self, style_img, gt_img):
        """simple matching"""
        mean = torch.mean(gt_img, dim=[1, 2])
        std = torch.std(gt_img, dim=[1, 2])

        mean_ = torch.mean(self.image_cache[style_img]["img"], dim=[1, 2])
        std_ = torch.std(self.image_cache[style_img]["img"], dim=[1, 2])

        normed_img = (gt_img - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        matched_img = normed_img * std_.view(3, 1, 1) + mean_.view(3, 1, 1)

        return matched_img

    def get_dino_feat(self, img):
        feats = {}

        dino_feat = self.dino_extractor(img, self.dino_extractor.get_input_size(img))
        for i in range(len(dino_feat)):
            feats["dino_cls_token_" + str(i)] = dino_feat[i][1][0]
            feats["dino_feature_map_" + str(i)] = dino_feat[i][0]

        return feats

    def get_vgg_feat(self, img):
        feats = {}

        vgg_feat = self.vgg_extractor(img, self.vgg_extractor.get_input_size(img))
        for i in range(len(vgg_feat)):
            feats["vgg_feature_map_" + str(i)] = vgg_feat[i]
            feats["vgg_mean_" + str(i)] = torch.mean(vgg_feat[i], dim=[0, 2, 3])
            feats["vgg_std_" + str(i)] = torch.std(vgg_feat[i], dim=[0, 2, 3])
            feats["vgg_gram_" + str(i)] = gram_matrix(vgg_feat[i])

        return feats

    def get_adain_style_loss(self, style_img, rendered_feat):
        # TODO 加上警告 assert style_img in self.image_cache
        style_feat = self.image_cache[style_img]

        adain_style_loss = 0.0
        for i in range(len(self.vgg_extractor.out_indices)):
            adain_style_loss += l2_loss(
                style_feat["vgg_gram_" + str(i)],
                rendered_feat["vgg_gram_" + str(i)],
            )
        #
        return 10 * adain_style_loss

    def get_adain_content_loss(self, style_img, rendered_feat):
        style_feat = self.image_cache[style_img]

        adain_content_loss = 0.0
        for i in range(len(self.vgg_extractor.out_indices)):
            adain_content_loss += l2_loss(
                style_feat["vgg_mean_" + str(i)],
                rendered_feat["vgg_mean_" + str(i)],
            )
            adain_content_loss += l2_loss(
                style_feat["vgg_std_" + str(i)],
                rendered_feat["vgg_std_" + str(i)],
            )

        return adain_content_loss

    def get_nn_style_loss(self, style_img, rendered_feat):
        style_feat = self.image_cache[style_img]

        nn_style_loss = 0.0
        for i in range(len(self.vgg_extractor.out_indices)):
            style_fmap = style_feat["vgg_feature_map_" + str(i)]
            rendered_fmap = rendered_feat["vgg_feature_map_" + str(i)]

            style_vec = style_fmap[0].view(style_fmap.shape[1], -1)
            rendered_vec = rendered_fmap[0].view(rendered_fmap.shape[1], -1)

            style_vec = style_vec / torch.norm(style_vec, dim=0, keepdim=True)
            rendered_vec = rendered_vec / torch.norm(rendered_vec, dim=0, keepdim=True)

            st = 0
            batch_size = 128**2
            nn_style_loss_ = 0.0
            while st < rendered_vec.shape[1]:
                tmp = 1.0 - rendered_vec[
                    :, st : min(st + batch_size, rendered_vec.shape[1])
                ].T.matmul(style_vec)

                nn, _ = torch.min(tmp, dim=1)
                nn_style_loss_ += torch.sum(nn)

                st += batch_size

            nn_style_loss += nn_style_loss_ / rendered_vec.shape[1]

        return nn_style_loss

    def get_rec_loss(self, rendered_img, content_img, l=0.2):
        """计算重建损失(content loss与rendered_img)"""
        Ll1 = l1_loss(rendered_img, content_img)
        ssim_loss = 1.0 - ssim(rendered_img, content_img)

        rec_loss = (1.0 - l) * Ll1 + l * ssim_loss

        return rec_loss

    def get_content_loss(self, rendered_feat, content_feat):
        content_loss = 0.0
        i = len(self.vgg_extractor.out_indices) - 1
        # for i in range(len(self.vgg_extractor.out_indices)):
        assert (
            content_feat["vgg_feature_map_" + str(i)].shape
            == rendered_feat["vgg_feature_map_" + str(i)].shape
        )
        content_loss += l2_loss(
            content_feat["vgg_feature_map_" + str(i)],
            rendered_feat["vgg_feature_map_" + str(i)],
        )

        return content_loss

    def predict(self, sh, opacity, scale_rot, style_img):
        """all sh use for prediction"""
        N = sh.shape[0]
        inp = sh.detach().view(sh.shape[0], -1).unsqueeze(0)

        # preprocess
        _sh = sh.detach().view(10 * N, -1)
        _sh = _sh.view(10 * N, 16, 3)
        _sh_0, _sh_1, _sh_2, _sh_3 = torch.split(_sh, [1, 3, 5, 7], dim=1)

        # inp = sh_0
        # inp = torch.cat([sh_0, sh_1, sh_2, sh_3], dim=1)

        # N = inp.shape[1]

        # sh_0_pred=[]
        # for i in range(0, N, batch_size):
        #     _inp = inp[:, i : min(i + batch_size, N), :]
        #     _inp = self.net["proj"](_inp)
        #     for i in range(len(self.dino_extractor.out_indices)):
        #         q = self.image_cache[style_img]["dino_feature_map_" + str(i)]
        #         q = q.permute(0, 2, 3, 1)
        #         q = q.view(1, -1, 384)
        #         _inp, _ = self.net["block_" + str(i)](_inp, q)

        #     _sh_0_pred = self.net["head"](_inp)

        #     _sh_0_pred = _sh_0_pred.permute(1, 0, 2)
        #     sh_0_pred.append(_sh_0_pred)

        inp = self.net["proj"](inp)
        # inp = inp.permute(1, 0, 2)

        for i in range(len(self.dino_extractor.out_indices)):
            q = self.image_cache[style_img]["dino_feature_map_" + str(i)]
            q = q.permute(0, 2, 3, 1)
            q = q.view(1, -1, 384)
            inp, _ = self.net["block_" + str(i)](inp, q)

        pred = self.net["head"](inp)[0]

        _sh_0_pred = pred.view(10 * N, 1, -1)
        out_sh = torch.cat([_sh_0 + _sh_0_pred, _sh_1, _sh_2, _sh_3], dim=1)
        out_sh = out_sh.view(10 * N, -1).view(N, -1)

        out_opacity = opacity
        out_scale_rot = scale_rot

        return out_sh, out_opacity, out_scale_rot

    # TODO save


def training(
    dataset,
    opt,
    pipe,
    dataset_name,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    wandb=None,
    logger=None,
    ply_path=None,
    # must-set
    output_path=None,
    style_img=None,
    # load_iteration=None,  # 理论上不用设置，一直是最后一个位置开始的
):
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianSHModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
    )
    gaussians.active_sh_degree = dataset.max_sh_degree  # all degree
    scene = SceneSH(
        dataset,
        gaussians,
        load_iteration=-1,
        # ply_path=ply_path,
        shuffle=False,
    )

    """为了隔离起见，不使用gaussian自带的optimizer进行训练;
    几何相关的pipeline也被禁用;
    logger使用自己重新实现的，用于监测style相关的"""
    # gaussians.training_setup(opt) # NOTE 和update_learning_rate 需要区分

    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)
    assert style_img is not None
    assert output_path is not None
    scene.model_path = output_path  # reset model_path
    model = StyleTransformer()  # 管理style transformer的模型
    model.update_image_cache(style_img)
    # VGG 特征实时计算/DINO特征预先缓存
    # model.training_setup(opt, gaussians)
    # extractor = VGG19Extractor(out_indices=[0, 2, 5, 7, 10])
    # extractor = VGG19Extractor(out_indices=[0, 5, 10, 19, 28])

    # load style img
    # target_img = cv2.imread(style_img)
    # target_img = torch.from_numpy(target_img).permute(2, 0, 1) / 255.0
    # target_img = target_img.unsqueeze(0).float().cuda()
    # """这部分重写"""
    # cam = scene.getTrainCameras()[0]
    # # all training images are same resolution by default
    # target_size = [cam.image_width, cam.image_height]
    # target_size = target_img.shape[-2:]
    # with torch.no_grad():
    #     target_features = extractor(target_img, target_size)
    #     target_grams = []  # C*C
    #     for target_feature in target_features:
    #         target_grams.append(gram_matrix(target_feature))

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # network gui not available in scaffold-gs yet
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         (
        #             custom_cam,
        #             do_training,
        #             pipe.convert_SHs_python,
        #             pipe.compute_cov3D_python,
        #             keep_alive,
        #             scaling_modifer,
        #         ) = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(
        #                 custom_cam, gaussians, pipe, background, scaling_modifer
        #             )["render"]
        #             net_image_bytes = memoryview(
        #                 (torch.clamp(net_image, min=0, max=1.0) * 255)
        #                 .byte()
        #                 .permute(1, 2, 0)
        #                 .contiguous()
        #                 .cpu()
        #                 .numpy()
        #             )
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and (
        #             (iteration < int(opt.iterations)) or not keep_alive
        #         ):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        # NOTE start here ignore network gui
        iter_start.record()

        # gaussians.update_learning_rate(iteration) # NOTE
        # model.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = iteration < opt.update_until and iteration >= 0

        render_pkg = render_sh(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
            transformer=model,  # for stylization
            style_img=style_img,
        )
        img = render_pkg["render"]
        # img=torch.clamp(img, min=0, max=1.0)
        gt_img = viewpoint_cam.original_image.cuda()
        with torch.no_grad():
            # content_img = model.color_matching(style_img, gt_img)
            content_img = model.color_matching_preserve(style_img, gt_img)
            content_feat = model.get_vgg_feat(content_img.unsqueeze(0))
            # content_img_ = model.color_matching_naiive(style_img, gt_img)
            # torchvision.utils.save_image(content_img, "./content_img.png")
            # torchvision.utils.save_image(content_img_preserve, "./content_img_.png")
            # import pdb
            # pdb.set_trace()

        feat = model.get_vgg_feat(img.unsqueeze(0))

        ### reconstruction loss
        rec_loss = model.get_rec_loss(img, content_img, l=opt.lambda_dssim)
        ### adain style loss
        adain_style_loss = model.get_adain_style_loss(style_img, feat)
        ## adain content loss
        adain_content_loss = model.get_adain_content_loss(style_img, feat)
        ### content loss
        content_loss = model.get_content_loss(feat, content_feat)
        ### nn style loss 会爆显存
        # nn_style_loss = model.get_nn_style_loss(style_img, feat)

        ### color matching ###

        # torchvision.utils.save_image(matched_image, "./matched_image.png")
        # torchvision.utils.save_image(image, "./image.png")

        # TODO manipulation
        # loss = rec_loss
        # loss = adain_style_loss + adain_content_loss + rec_loss
        # loss = rec_loss + nn_style_loss + content_loss
        loss = adain_style_loss + adain_content_loss + rec_loss + content_loss
        # loss = style_loss + content_loss
        # scaling_reg = scaling.prod(dim=1).mean()
        # loss = (
        #     (1.0 - opt.lambda_dssim) * Ll1
        #     + opt.lambda_dssim * ssim_loss
        #     + 0.01 * scaling_reg
        # )
        if iteration % 100 == 0:
            print("adain_style_loss:", adain_style_loss.item())
            print("adain_content_loss:", adain_content_loss.item())
            print("content_loss:", content_loss.item())
            # print("nn_style_loss:", nn_style_loss.item())
            print("rec_loss:", rec_loss.item())
            torchvision.utils.save_image(img, f"./img_{iteration}.png")

        loss.backward()

        iter_end.record()

        # 记录数据
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(
            #     tb_writer,
            #     dataset_name,
            #     iteration,
            #     Ll1,
            #     loss,
            #     l1_loss,
            #     iter_start.elapsed_time(iter_end),
            #     testing_iterations,
            #     scene,
            #     render_sh,
            #     (pipe, background),
            #     wandb,
            #     logger,
            # )
            if iteration in saving_iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # densification
            # if iteration < opt.update_until and iteration > opt.start_stat:
            #     # add statis
            #     gaussians.training_statis(
            #         viewspace_point_tensor,
            #         opacity,
            #         visibility_filter,
            #         offset_selection_mask,
            #         voxel_visible_mask,
            #     )

            #     # densification
            #     if iteration > opt.update_from and iteration % opt.update_interval == 0:
            #         gaussians.adjust_anchor(
            #             check_interval=opt.update_interval,
            #             success_threshold=opt.success_threshold,
            #             grad_threshold=opt.densify_grad_threshold,
            #             min_opacity=opt.min_opacity,
            #         )
            # elif iteration == opt.update_until:
            #     del gaussians.opacity_accum
            #     del gaussians.offset_gradient_accum
            #     del gaussians.offset_denom
            #     torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

            """ckpt存储方式，这里依然使用scene的管理方式"""
            # if iteration in checkpoint_iterations:
            #
            #     logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save(
            #         (gaussians.capture(), iteration),
            #         scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            #     )

    # save
    model.save_model(output_path)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# def training_report(
#     tb_writer,
#     dataset_name,
#     iteration,
#     Ll1,
#     loss,
#     l1_loss,
#     elapsed,
#     testing_iterations,
#     scene: Scene,
#     renderFunc,
#     renderArgs,
#     wandb=None,
#     logger=None,
# ):
#     if tb_writer:
#         tb_writer.add_scalar(
#             f"{dataset_name}/train_loss_patches/l1_loss", Ll1.item(), iteration
#         )
#         tb_writer.add_scalar(
#             f"{dataset_name}/train_loss_patches/total_loss", loss.item(), iteration
#         )
#         tb_writer.add_scalar(f"{dataset_name}/iter_time", elapsed, iteration)

#     if wandb is not None:
#         wandb.log(
#             {
#                 "train_l1_loss": Ll1,
#                 "train_total_loss": loss,
#             }
#         )

#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         scene.gaussians.eval()
#         torch.cuda.empty_cache()
#         validation_configs = (
#             {"name": "test", "cameras": scene.getTestCameras()},
#             {
#                 "name": "train",
#                 "cameras": [
#                     scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
#                     for idx in range(5, 30, 5)
#                 ],
#             },
#         )

#         for config in validation_configs:
#             if config["cameras"] and len(config["cameras"]) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0

#                 if wandb is not None:
#                     gt_image_list = []
#                     render_image_list = []
#                     errormap_list = []

#                 for idx, viewpoint in enumerate(config["cameras"]):
#                     voxel_visible_mask = prefilter_voxel(
#                         viewpoint, scene.gaussians, *renderArgs
#                     )
#                     image = torch.clamp(
#                         renderFunc(
#                             viewpoint,
#                             scene.gaussians,
#                             *renderArgs,
#                             visible_mask=voxel_visible_mask,
#                         )["render"],
#                         0.0,
#                         1.0,
#                     )
#                     gt_image = torch.clamp(
#                         viewpoint.original_image.to("cuda"), 0.0, 1.0
#                     )
#                     if tb_writer and (idx < 30):
#                         tb_writer.add_images(
#                             f"{dataset_name}/"
#                             + config["name"]
#                             + "_view_{}/render".format(viewpoint.image_name),
#                             image[None],
#                             global_step=iteration,
#                         )
#                         tb_writer.add_images(
#                             f"{dataset_name}/"
#                             + config["name"]
#                             + "_view_{}/errormap".format(viewpoint.image_name),
#                             (gt_image[None] - image[None]).abs(),
#                             global_step=iteration,
#                         )

#                         if wandb:
#                             render_image_list.append(image[None])
#                             errormap_list.append((gt_image[None] - image[None]).abs())

#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(
#                                 f"{dataset_name}/"
#                                 + config["name"]
#                                 + "_view_{}/ground_truth".format(viewpoint.image_name),
#                                 gt_image[None],
#                                 global_step=iteration,
#                             )
#                             if wandb:
#                                 gt_image_list.append(gt_image[None])

#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()

#                 psnr_test /= len(config["cameras"])
#                 l1_test /= len(config["cameras"])
#                 logger.info(
#                     "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
#                         iteration, config["name"], l1_test, psnr_test
#                     )
#                 )

#                 if tb_writer:
#                     tb_writer.add_scalar(
#                         f"{dataset_name}/"
#                         + config["name"]
#                         + "/loss_viewpoint - l1_loss",
#                         l1_test,
#                         iteration,
#                     )
#                     tb_writer.add_scalar(
#                         f"{dataset_name}/" + config["name"] + "/loss_viewpoint - psnr",
#                         psnr_test,
#                         iteration,
#                     )
#                 if wandb is not None:
#                     wandb.log(
#                         {
#                             f"{config['name']}_loss_viewpoint_l1_loss": l1_test,
#                             f"{config['name']}_PSNR": psnr_test,
#                         }
#                     )

#         if tb_writer:
#             # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar(
#                 f"{dataset_name}/" + "total_points",
#                 scene.gaussians.get_anchor.shape[0],
#                 iteration,
#             )
#         torch.cuda.empty_cache()

#         scene.gaussians.train()


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    model,
    style_img,
    enable_depth=False,  # 记得关
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    # makedirs(error_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    if enable_depth:
        depth_path = os.path.join(
            model_path, name, "ours_{}".format(iteration), "depths"
        )
        makedirs(depth_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render_sh(
            view,
            gaussians,
            pipeline,
            background,
            visible_mask=voxel_visible_mask,
            transformer=model,  # for stylization
            style_img=style_img,
        )
        torch.cuda.synchronize()
        t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # # gts
        # gt = view.original_image[0:3, :, :]

        # # error maps
        # errormap = (rendering - gt).abs()

        # for depth
        if enable_depth:
            depth = render_depth_sh(
                view, gaussians, pipeline, background, visible_mask=voxel_visible_mask
            )["render"]
            # normalization
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            torchvision.utils.save_image(
                depth, os.path.join(depth_path, "{0:05d}".format(idx) + ".png")
            )

        name_list.append("{0:05d}".format(idx) + ".png")
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        # torchvision.utils.save_image(
        #     errormap, os.path.join(error_path, "{0:05d}".format(idx) + ".png")
        # )
        # torchvision.utils.save_image(
        #     gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        # )
        per_view_dict["{0:05d}".format(idx) + ".png"] = visible_count.item()

    with open(
        os.path.join(
            model_path, name, "ours_{}".format(iteration), "per_view_count.json"
        ),
        "w",
    ) as fp:
        json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train=True,
    skip_test=False,
    wandb=None,
    tb_writer=None,
    dataset_name=None,
    logger=None,
    output_path=None,
    style_img=None,
):
    with torch.no_grad():
        gaussians = GaussianSHModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
        )
        scene = SceneSH(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.active_sh_degree = dataset.max_sh_degree  # all degree
        gaussians.eval()

        assert style_img is not None
        assert output_path is not None
        scene.model_path = output_path  # reset model_path
        model = StyleTransformer()
        # 加载模型
        model.update_image_cache(style_img)
        model.load_model(output_path)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count = render_set(
                output_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                model,
                style_img,
            )
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f"Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m")
            if wandb is not None:
                wandb.log(
                    {
                        "train_fps": train_fps.item(),
                    }
                )

        if not skip_test:
            t_test_list, visible_count = render_set(
                output_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                model,
                style_img,
            )
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f"Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m")
            if tb_writer:
                tb_writer.add_scalar(f"{dataset_name}/test_FPS", test_fps.item(), 0)
            if wandb is not None:
                wandb.log(
                    {
                        "test_fps": test_fps,
                    }
                )

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(
    model_paths,
    visible_count=None,
    wandb=None,
    tb_writer=None,
    dataset_name=None,
    logger=None,
):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        if wandb is not None:
            wandb.log(
                {
                    "test_SSIMS": torch.stack(ssims).mean().item(),
                }
            )
            wandb.log(
                {
                    "test_PSNR_final": torch.stack(psnrs).mean().item(),
                }
            )
            wandb.log(
                {
                    "test_LPIPS": torch.stack(lpipss).mean().item(),
                }
            )

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info(
            "  SSIM : \033[1;35m{:>12.7f}\033[0m".format(
                torch.tensor(ssims).mean(), ".5"
            )
        )
        logger.info(
            "  PSNR : \033[1;35m{:>12.7f}\033[0m".format(
                torch.tensor(psnrs).mean(), ".5"
            )
        )
        logger.info(
            "  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(
                torch.tensor(lpipss).mean(), ".5"
            )
        )
        print("")

        if tb_writer:
            tb_writer.add_scalar(
                f"{dataset_name}/SSIM", torch.tensor(ssims).mean().item(), 0
            )
            tb_writer.add_scalar(
                f"{dataset_name}/PSNR", torch.tensor(psnrs).mean().item(), 0
            )
            tb_writer.add_scalar(
                f"{dataset_name}/LPIPS", torch.tensor(lpipss).mean().item(), 0
            )

            tb_writer.add_scalar(
                f"{dataset_name}/VISIBLE_NUMS",
                torch.tensor(visible_count).mean().item(),
                0,
            )

        full_dict[scene_dir][method].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict[scene_dir][method].update(
            {
                "SSIM": {
                    name: ssim
                    for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                },
                "PSNR": {
                    name: psnr
                    for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                },
                "LPIPS": {
                    name: lp
                    for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                },
                "VISIBLE_COUNT": {
                    name: vc
                    for vc, name in zip(
                        torch.tensor(visible_count).tolist(), image_names
                    )
                },
            }
        )

    with open(scene_dir + "/results.json", "w") as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", "w") as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="-1")
    # style utils
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--style_img", type=str)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  # 结束一定会保存

    # enable logging

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f"args: {args}")

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f"using GPU {args.gpu}")

    # save memory !!!
    # try:
    #     saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')

    dataset = args.source_path.split("/")[-1]
    exp_name = args.model_path.split("/")[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args),
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        dataset,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        wandb,
        logger,
        output_path=args.output_path,
        style_img=args.style_img,
    )

    # if args.warmup:
    #     logger.info("\n Warmup finished! Reboot from last checkpoints")
    #     new_ply_path = os.path.join(
    #         args.model_path,
    #         f"point_cloud/iteration_{args.iterations}",
    #         "point_cloud.ply",
    #     )
    #     training(
    #         lp.extract(args),
    #         op.extract(args),
    #         pp.extract(args),
    #         dataset,
    #         args.test_iterations,
    #         args.save_iterations,
    #         args.checkpoint_iterations,
    #         args.start_checkpoint,
    #         args.debug_from,
    #         wandb=wandb,
    #         logger=logger,
    #         ply_path=new_ply_path,
    #     )

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f"\nStarting Rendering~")
    # visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    visible_count = render_sets(
        lp.extract(args),
        -1,
        pp.extract(args),
        wandb=wandb,
        logger=logger,
        skip_train=False,  # only render train set
        skip_test=True,
        output_path=args.output_path,
        style_img=args.style_img,
    )
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    # TODO
    # evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
