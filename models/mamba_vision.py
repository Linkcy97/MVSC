#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import random
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from .channel import Channel


def window_partition(x, window_size):
    """ccccc
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)   
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)  # bs, ws, pix in ws, C
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 method = 'conv'
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.method = method
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        if self.method == 'conv':
            self.reduction = nn.Sequential(
                nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
            )
        else:
            self.norm = nn.LayerNorm(4 * dim)
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            

    def forward(self, x):
        if self.method == 'conv':
            x = self.reduction(x)         # bs,C,H,W    ---> bs,2*C,H/2,W/2
        else:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b h w c')
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C
            x = self.norm(x)
            x = self.reduction(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H//2, w=W//2)
        return x

class Upsample(nn.Module):
    """
    Up-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 method='PixelShuffle',
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.method = method
        if keep_dim:
            dim_out = 3
        else:
            dim_out = dim // 2
        if method == 'bilinear':
            pre_upsample = nn.Conv2d(dim, dim_out, 3, 1, 1, bias=False)
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif method == 'convTranspose':
            pre_upsample = nn.Identity()
            upsample = nn.ConvTranspose2d(dim, dim_out, 3, 2, 1, 1, bias=False)
        elif method == 'PixelShuffle':
            pre_upsample = nn.Conv2d(dim, dim_out * 4, 3, 1, 1, bias=False)
            upsample = nn.PixelShuffle(2)
        elif method == 'patch':
            self.pre_upsample = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_out * 4, bias=False),
            )
            self.upsample = nn.PixelShuffle(2)
        if method != 'patch':
            self.expansion = nn.Sequential(
                pre_upsample,
                upsample
            )

    def forward(self, x):
        if self.method != 'patch':
            x = self.expansion(x)
        else:
            _, _, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.pre_upsample(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            x = self.upsample(x)

        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block "stem" in the paper
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        # self.conv_down = nn.Sequential(
        #     nn.Conv2d(in_chans, in_dim, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(in_dim, eps=1e-4),
        #     nn.Sigmoid(),
        #     nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(dim, eps=1e-4),
        #     nn.Sigmoid()
        #     )
        self.conv_down = nn.Sequential(
                    nn.Conv2d(in_chans, dim, 2, 2)
                    )
    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class PatchUnembed(nn.Module):
    """
    Inverse operation of PatchEmbed to restore original dimensions.
    """

    def __init__(self, out_chans=3, dim=96, out_dim=64):
        """
        Args:
            out_chans: number of output channels, should match the original input channels of PatchEmbed.
            dim: the intermediate feature size dimension, should match the 'dim' in PatchEmbed.
            out_dim: the dimension after the first transposed convolution, should match 'in_dim' in PatchEmbed.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(dim, out_dim, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_chans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_up(x)
        x = self.proj(x)
        return x
    
class DenoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cnn_denoise = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        noise = self.cnn_denoise(x)
        noise = self.norm(noise)
        noise = self.act(noise)
        return x + noise

class SnrEstimate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim*16, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.resnet1 = nn.Sequential(
            nn.Conv2d(dim*16, dim*16, (1,3), 1, 'same'),
            # nn.BatchNorm2d(dim*16),
            nn.LeakyReLU(),
            nn.Conv2d(dim*16, dim*16, (1,3), 1, 'same'),
            # nn.BatchNorm2d(dim*16),
            nn.LeakyReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1,32))
        self.fc = nn.Linear(16*32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.resnet1(x)
        x = x + x1
        x = self.avgpool(x)
        x = rearrange(x, 'b c h w -> b (h c w)')
        x = self.fc(x)
        return x
    
class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x                       # (bs,C,H,W) --->(bs,C,H,W)


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)               # remain size
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=False,
                 upsample=False,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
                 net_type = 'encoder'
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.transformer_block = True
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale,)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim,method='PixelShuffle')
        self.upsample = None if not upsample else Upsample(dim=dim,method='patch')
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size 
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:                         # if it is not divisible, add 0
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):        # first two are conv, another are block
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

        if self.downsample is None and self.upsample is None:
            return x
        elif self.downsample:
            return self.downsample(x)
        elif self.upsample:
            return self.upsample(x)


class MambaVisionEncoder(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 C,
                 drop_path_rate=0.2,
                 in_chans=3,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i < int(len(depths)/2)) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < (len(depths) - 1)),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     net_type = 'encoder'
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.head_c = nn.Linear(num_features, C)
        self.tanh = nn.Tanh()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)       # (bs,3,H,W)   --> (bs,C,H/2,W/2)
        for level in self.levels:     # depth of MambaVisionLayer the first half of depth is conv
            x = level(x)
        x = self.norm(x)              # (bs,8*C,H/32,W/32)
        x_h = x.shape[2]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.head_c(x)
        # x = self.tanh(x)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=x_h)
        return x,x_h

    def forward(self, x):
        x = self.forward_features(x)
        return x
        

class MambaVisionDecoder(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 C,
                 drop_path_rate=0.2,
                 in_chans=3,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim)
        self.patch_unembed = PatchUnembed(out_chans=in_chans, out_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths) - 1, -1, -1):
            conv = True if (i < int(len(depths)/2)) else False
            denosie = DenoiseBlock(dim=int(dim * 2 ** i))
            self.levels.append(denosie)
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     upsample=(i > 0),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     net_type = 'decoder'
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.head_c = nn.Linear(2*C, int(dim * 2 ** (len(depths) - 1)))
        self.classfiy_net = nn.Sequential(
            nn.Conv2d(dim, dim//2, 4, 4),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(),
            nn.Conv2d(dim//2, 10, 4, 4),
            nn.BatchNorm2d(10),
            nn.Sigmoid())
        # self.snr_net = nn.Sequential(
        #     nn.Conv2d(dim, dim//2, 4, 4),
        #     nn.BatchNorm2d(dim//2),
        #     nn.ReLU(),
        #     nn.Conv2d(dim//2, 5, 4, 4),
        #     nn.BatchNorm2d(5),
        #     nn.Sigmoid())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x, x_h):
        # x_h = x.shape[2]
        # x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.head_c(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=x_h)
        for level in self.levels:     
            x = level(x)

        x = self.norm(x) 
        cla = self.classfiy_net(x)
        cla = rearrange(cla, 'b c h w -> b (h w c)')
        # snr = self.snr_net(x)
        # snr = rearrange(snr, 'b c h w -> b (h w c)')
        x =  self.patch_unembed(x)

        return x, cla

    def forward(self, x, x_h):
        x, cla = self.forward_features(x, x_h)
        return x, cla


class MVSC(nn.Module):

    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.encoder = MambaVisionEncoder(dim=config.model_config['dim'],
                                            in_dim=config.model_config['in_dim'],
                                            C=config.model_config['C'],
                                            depths=config.model_config['depths'],
                                            window_size=config.model_config['window_size'],
                                            mlp_ratio=config.model_config['mlp_ratio'],
                                            num_heads=config.model_config['num_heads'],
                                            drop_path_rate=config.model_config['drop_path_rate'],
                                          )
        self.decoder = MambaVisionDecoder(dim=config.model_config['dim'],
                                            in_dim=config.model_config['in_dim'],
                                            C=config.model_config['C'],
                                            depths=config.model_config['depths'],
                                            window_size=config.model_config['window_size'],
                                            mlp_ratio=config.model_config['mlp_ratio'],
                                            num_heads=config.model_config['num_heads'],
                                            drop_path_rate=config.model_config['drop_path_rate'],
                                          )
        self.channel = Channel(config)
        # self.multiple_snr = [1, 4, 7, 10, 13]    nn.Conv2d(8, 8, 3, 1, 1)
        self.cnn_denoise = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.snr_est = SnrEstimate(1)
                                     
        self.multiple_snr = config.multiple_snr


    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False


    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    def freeze_snr(self):
        for param in self.cnn_denoise.parameters():
            param.requires_grad = False   

    def unfreeze_snr(self):
        for param in self.cnn_denoise.parameters():
            param.requires_grad = True   


    def forward(self, x, given_snr=False):
        semantic_feature, x_h = self.encoder(x)
        CBR = semantic_feature.numel() / 2 / x.numel()
        if given_snr:
            g_snr = given_snr
            choice = self.multiple_snr.index(g_snr)
        else:
            choice = random.randint(0, len(self.multiple_snr) - 1)
            g_snr = self.multiple_snr[choice]
        # ones = torch.ones_like(semantic_feature)
        x_noise = self.channel(semantic_feature, g_snr)
        # x_noise1 = x_noise - ones
        # x_noise1 = rearrange(x_noise, 'b hw c -> b (hw c)')
        # x_noise1 = rearrange(x_noise1, 'b (c h w) -> b c h w', c=1,h=2)

        # snr = self.snr_est(x_noise1)
        # x_ded = self.liner_denoise(x_noise)
        # x_ded = x_noise + x_de
        x_noise = rearrange(x_noise, 'b (h w) c -> b c h w',h=x_h)
        noise = self.cnn_denoise(x_noise)
        x_signal = x_noise + noise
        snr =10*torch.log10(torch.mean(x_signal**2, dim=[1, 2, 3]) / torch.mean(noise**2, dim=[1, 2, 3]))
        # x_ded = x_noise + x_de
        x_noise = rearrange(x_noise, 'b c h w -> b (h w) c')
        noise = rearrange(noise, 'b c h w -> b (h w) c')
        x_denoise = torch.cat((x_noise, noise), dim=2)
        x, cla = self.decoder(x_denoise,x_h)
        return x, CBR, g_snr, snr, cla
