# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:56:00 2022

@author: AruZeng
"""
import math
#from torchsummary import summary

#from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#
NEG_INF = -1000000
#
#
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=64, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // 4, img_size[1] // 4] # only for flops calculation
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - 1) // 2
            self.projs.append(nn.Conv3d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W,D = x.shape
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=1)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                dim = self.embed_dim // 2 ** i
            else:
                dim = self.embed_dim // 2 ** (i + 1)
            flops += Ho * Wo * dim * self.in_chans * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
#
#
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#
class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(3, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class Attention(nn.Module):
    r""" Multi-head self attention module with relative position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W,D, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Gh*Gw, Gh*Gw) or None
        """
        group_size = (H, W,D)
        B_, N, C = x.shape
        assert H*W*D == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            position_bias_d = torch.arange(1 - group_size[2], group_size[2], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w,position_bias_d]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_d = torch.arange(group_size[0])
            coords_h = torch.arange(group_size[1])
            coords_w = torch.arange(group_size[2])
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 2] += group_size[2] - 1

            relative_coords[:, :, 0] *= (2 * group_size[1] - 1) * (2 * group_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * group_size[2] - 1)
            relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

            pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view( 
                group_size[0] * group_size[1]*group_size[2], group_size[0] * group_size[1]*group_size[2], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            #print(attn.shape)
            #print(relative_position_bias.shape)
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nG = mask.shape[0]
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        excluded_flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        excluded_flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        excluded_flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops, excluded_flops
#
#
#
class CrossAttention(nn.Module):
    r""" Multi-head self attention module with relative position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y=nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,y, H, W,D, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Gh*Gw, Gh*Gw) or None
        """
        group_size = (H, W,D)
        B_, N, C = x.shape
        assert H*W*D == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        qkv_y = self.qkv(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        #
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q_y,k_y,v_y=qkv_y[0], qkv_y[1], qkv_y[2]
        #
        q = q * self.scale
        attn = (q @ k_y.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            position_bias_d = torch.arange(1 - group_size[2], group_size[2], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w,position_bias_d]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_d = torch.arange(group_size[0])
            coords_h = torch.arange(group_size[1])
            coords_w = torch.arange(group_size[2])
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 2] += group_size[2] - 1

            relative_coords[:, :, 0] *= (2 * group_size[1] - 1) * (2 * group_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * group_size[2] - 1)
            relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

            pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view( 
                group_size[0] * group_size[1]*group_size[2], group_size[0] * group_size[1]*group_size[2], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            #print(attn.shape)
            #print(relative_position_bias.shape)
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nG = mask.shape[0]
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v_y).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        excluded_flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        excluded_flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        excluded_flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops, excluded_flops
#
#
#
class CrossFormerBlock(nn.Module):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Window size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, rev, group_size=4, interval=8, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        self.rev=rev
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W,D):
        #print(x.shape)
        B, L, C = x.shape
        assert L == H * W*D, "input feature has wrong size %d, %d, %d,%d" % (L, H, W, D)

        if min(H, W,D) <= self.group_size:
            # if window size is larger than input resolution, we don't partition windows
            self.lsda_flag = 0
            self.group_size = min(H, W,D)

        shortcut = x
        #print(shortcut.shape)
        x = self.norm1(x)
        x = x.view(B, H, W,D, C)

        # padding
        size_div = self.interval if self.lsda_flag == 1 else self.group_size
        pad_l = pad_t = pad_q= 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        pad_d = (size_div - D % size_div) % size_div
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b,pad_q,pad_d))
        _, Hp, Wp, Dp,_ = x.shape

        mask = torch.zeros((1, Hp, Wp,Dp, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1
        if pad_d > 0:
            mask[:, :, -pad_d:, :] = -1

        # group embeddings and generate attn_mask
        if self.lsda_flag == 0: # SDA
            G = Gh = Gw = Gd = self.group_size
            x = x.reshape(B, Hp // G, G, Wp // G, G,Dp//G,G, C).permute(0, 1, 3,5, 2, 4,6, 7).contiguous()
            x = x.reshape(B * Hp * Wp*Dp // G**3, G**3, C)
            nG = Hp * Wp * Dp // G**3
            # attn_mask
            if pad_r > 0 or pad_b > 0 or pad_d>0:
                mask = mask.reshape(1, Hp // G, G, Wp // G, G, Dp//G,G ,1).permute(0, 1, 3,5, 2, 4,6, 7).contiguous()
                mask = mask.reshape(nG, 1, G * G * G)
                attn_mask = torch.zeros((nG, G * G *G, G * G*G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        else: # LDA
            I, Gh, Gw ,Gd = self.interval, Hp // self.interval, Wp // self.interval, Dp//self.interval
            x = x.reshape(B, Gh, I, Gw, I,Gd,I, C).permute(0, 2, 4,6, 1, 3, 5,7).contiguous()
            x = x.reshape(B * I * I*I, Gh * Gw * Gd, C)
            nG = I ** 3
            # attn_mask
            if pad_r > 0 or pad_b > 0 or pad_d>0:
                mask = mask.reshape(1, Gh, I, Gw, I,Gd,I, 1).permute(0, 2, 4,6, 1, 3, 5,7).contiguous()
                mask = mask.reshape(nG, 1, Gh * Gw * Gd)
                attn_mask = torch.zeros((nG, Gh * Gw * Gd, Gh * Gw * Gd), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # multi-head self-attention
        x = self.attn(x, Gh, Gw, Gd,mask=attn_mask)  # nG*B, G*G, C
        
        # ungroup embeddings
        if self.lsda_flag == 0:
            x = x.reshape(B, Hp // G, Wp // G, Dp//G, G, G, G, C).permute(0, 1, 4, 2, 5,3,6, 7).contiguous() # B, Hp//G, G, Wp//G, G, C
        else:
            x = x.reshape(B, I, I,I, Gh, Gw,Gd, C).permute(0, 4, 1, 5, 2,6,3, 7).contiguous() # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hp, Wp,Dp, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :D, :].contiguous()
        x = x.view(B, H * W * D, C)

        # FFN
        #print(x.shape)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # Attention
        size_div = self.interval if self.lsda_flag == 1 else self.group_size
        Hp = math.ceil(H / size_div) * size_div
        Wp = math.ceil(W / size_div) * size_div
        Gh = Hp / size_div if self.lsda_flag == 1 else self.group_size
        Gw = Wp / size_div if self.lsda_flag == 1 else self.group_size
        nG = Hp * Wp / Gh / Gw
        attn_flops, attn_excluded_flops = self.attn.flops(Gh * Gw)
        flops += nG * attn_flops
        excluded_flops = nG * attn_excluded_flops
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops, excluded_flops
#
#
class CrossAttentionFormerBlock(nn.Module):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Window size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, rev, group_size=4, interval=4, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        self.rev=rev
        self.norm1 = norm_layer(dim)

        self.attn = CrossAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,y, H, W,D):
        #print(x.shape)
        B, L, C = x.shape
        assert L == H * W*D, "input feature has wrong size %d, %d, %d,%d" % (L, H, W, D)

        if min(H, W,D) <= self.group_size:
            # if window size is larger than input resolution, we don't partition windows
            self.lsda_flag = 0
            self.group_size = min(H, W,D)

        shortcut = x
        #print(shortcut.shape)
        x = self.norm1(x)
        x = x.view(B, H, W,D, C)
        y = y.view(B,H,W,D,C)
        # padding
        size_div = self.interval if self.lsda_flag == 1 else self.group_size
        pad_l = pad_t = pad_q= 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        pad_d = (size_div - D % size_div) % size_div
        #
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b,pad_q,pad_d))
        y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b,pad_q,pad_d))
        #
        _, Hp, Wp, Dp,_ = x.shape

        mask = torch.zeros((1, Hp, Wp,Dp, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1
        if pad_d > 0:
            mask[:, :, -pad_d:, :] = -1

        # group embeddings and generate attn_mask
        if self.lsda_flag == 0: # SDA
            G = Gh = Gw = Gd = self.group_size
            x = x.reshape(B, Hp // G, G, Wp // G, G,Dp//G,G, C).permute(0, 1, 3,5, 2, 4,6, 7).contiguous()
            x = x.reshape(B * Hp * Wp*Dp // G**3, G**3, C)
            #
            y = y.reshape(B, Hp // G, G, Wp // G, G,Dp//G,G, C).permute(0, 1, 3,5, 2, 4,6, 7).contiguous()
            y = y.reshape(B * Hp * Wp*Dp // G**3, G**3, C)
            #
            nG = Hp * Wp * Dp // G**3
            # attn_mask
            if pad_r > 0 or pad_b > 0 or pad_d>0:
                mask = mask.reshape(1, Hp // G, G, Wp // G, G, Dp//G,G ,1).permute(0, 1, 3,5, 2, 4,6, 7).contiguous()
                mask = mask.reshape(nG, 1, G * G * G)
                attn_mask = torch.zeros((nG, G * G *G, G * G*G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        else: # LDA
            I, Gh, Gw ,Gd = self.interval, Hp // self.interval, Wp // self.interval, Dp//self.interval
            x = x.reshape(B, Gh, I, Gw, I,Gd,I, C).permute(0, 2, 4,6, 1, 3, 5,7).contiguous()
            x = x.reshape(B * I * I*I, Gh * Gw * Gd, C)
            #
            y = y.reshape(B, Gh, I, Gw, I,Gd,I, C).permute(0, 2, 4,6, 1, 3, 5,7).contiguous()
            y = y.reshape(B * I * I*I, Gh * Gw * Gd, C)
            #
            nG = I ** 3
            # attn_mask
            if pad_r > 0 or pad_b > 0 or pad_d>0:
                mask = mask.reshape(1, Gh, I, Gw, I,Gd,I, 1).permute(0, 2, 4,6, 1, 3, 5,7).contiguous()
                mask = mask.reshape(nG, 1, Gh * Gw * Gd)
                attn_mask = torch.zeros((nG, Gh * Gw * Gd, Gh * Gw * Gd), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # multi-head self-attention
        x = self.attn(x,y, Gh, Gw, Gd,mask=attn_mask)  # nG*B, G*G, C
        
        # ungroup embeddings
        if self.lsda_flag == 0:
            x = x.reshape(B, Hp // G, Wp // G, Dp//G, G, G, G, C).permute(0, 1, 4, 2, 5,3,6, 7).contiguous()# B, Hp//G, G, Wp//G, G, C
        else:
            x = x.reshape(B, I, I,I, Gh, Gw,Gd, C).permute(0, 4, 1, 5, 2,6,3, 7).contiguous() # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hp, Wp,Dp, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :D, :].contiguous()
        x = x.view(B, H * W * D, C)

        # FFN
        #print(x.shape)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # Attention
        size_div = self.interval if self.lsda_flag == 1 else self.group_size
        Hp = math.ceil(H / size_div) * size_div
        Wp = math.ceil(W / size_div) * size_div
        Gh = Hp / size_div if self.lsda_flag == 1 else self.group_size
        Gw = Wp / size_div if self.lsda_flag == 1 else self.group_size
        nG = Hp * Wp / Gh / Gw
        attn_flops, attn_excluded_flops = self.attn.flops(Gh * Gw)
        flops += nG * attn_flops
        excluded_flops = nG * attn_excluded_flops
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops, excluded_flops
#
#
class TwoCrossFormerBlocks(nn.Module):
    def __init__(self, dim, input_resolution):
        super().__init__()
        self.resolution=input_resolution
        self.block1=CrossFormerBlock(dim,input_resolution,4,rev=True,lsda_flag=0)
        self.block2=CrossFormerBlock(dim,input_resolution,4,rev=False,lsda_flag=1)
    def forward(self,x):
        #print(x.shape)
        _,_,H,W,D=x.shape
        x=rearrange(x,'B C H W D -> B (H W D) C',H=H,W=W,D=D)
        x=self.block1(x,H,W,D)
        x=self.block2(x,H,W,D)
        x=rearrange(x,'B (H W D) C -> B C H W D',H=H,W=W,D=D)
        return x
#
class CrossAttnFormerBlocks(nn.Module):
    def __init__(self, dim, input_resolution):
        super().__init__()
        self.resolution=input_resolution
        self.block1=CrossAttentionFormerBlock(dim,input_resolution,4,rev=True,lsda_flag=0)
        self.block2=CrossAttentionFormerBlock(dim,input_resolution,4,rev=False,lsda_flag=1)
        self.block3=CrossAttentionFormerBlock(dim,input_resolution,4,rev=True,lsda_flag=0)
    def forward(self,x,y):
        _,_,H,W,D=x.shape
        x=rearrange(x,'B C H W D -> B (H W D) C',H=H,W=W,D=D)
        y=rearrange(y,'B C H W D -> B (H W D) C',H=H,W=W,D=D)
        x1=self.block1(x,y,H,W,D)
        x2=self.block2(x,y,H,W,D)
        x=self.block3(x1,x2,H,W,D)
        x=rearrange(x,'B (H W D) C -> B C H W D',H=H,W=W,D=D)
        return x   
#
class Generator(nn.Module):#A Small Version which is more Memory Friendly
    def __init__(self):
        super().__init__()
        #下采样部分
        #layer0 1->64
        #64*64*64
        self.encoder_layer0_down=nn.Sequential(
            PatchEmbed(img_size=64,patch_size=[3,5,7],in_chans=1,embed_dim=64),
            #TwoCrossFormerBlocks(64,32),
            )#first conv embed
        self.encoder_layer0_mri = nn.Sequential(
            PatchEmbed(img_size=64, patch_size=[3, 5, 7], in_chans=1, embed_dim=64),
        )
        self.fuse0=CrossAttnFormerBlocks(64,32)
        #32*32*32
        self.encoder_layer1_down = nn.Sequential(
            PatchEmbed(img_size=32, patch_size=[3, 5, 7], in_chans=64, embed_dim=128),
            TwoCrossFormerBlocks(128, 16),
        )  # first conv embed
        self.encoder_layer1_mri = nn.Sequential(
            PatchEmbed(img_size=32, patch_size=[3, 5, 7], in_chans=64, embed_dim=128),
        )
        self.fuse1 = CrossAttnFormerBlocks(128, 16)
        #4*4*4
        self.encoder_layer2_down = nn.Sequential(
            PatchEmbed(img_size=16, patch_size=[3, 5, 7], in_chans=128, embed_dim=192),
            TwoCrossFormerBlocks(192, 8),
        )  # first conv embed
        self.encoder_layer2_mri = nn.Sequential(
            PatchEmbed(img_size=16, patch_size=[3, 5, 7], in_chans=128, embed_dim=192),
        )
        self.fuse2 = CrossAttnFormerBlocks(192, 8)
        #
        self.encoder_layer3_down=nn.Conv3d(192,256,kernel_size=3,stride=2,padding=1)
        #
        #下采样输出
        #
        #上采样部分
        '''
        self.decoder_layer1=nn.Sequential(
            nn.ConvTranspose3d(512*2,512,kernel_size=2,stride=1,padding=1),
            nn.ConvTranspose3d(512,384,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2),)
        self.transres1=ResBlock3D(1024, 384)
        self.decoder_layer1_up=nn.Sequential(
            nn.ConvTranspose3d(384, 256, kernel_size)
            )
        '''
        #4*4*4
        self.decoder_layer1=nn.Sequential(
            nn.Conv3d(256,256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(256),
            )
        #self.resup1=ResBlock3D(512, 512)
        self.decoder_layer1_up=nn.ConvTranspose3d(256,192,kernel_size=2,stride=2)
        #8*8*8
        self.decoder_layer2=nn.Sequential(
            nn.Conv3d(192*2,192,kernel_size=3,stride=1,padding=1),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(192),
            )
        #self.resup2=ResBlock3D(256*2, 256)
        self.decoder_layer2_up=nn.ConvTranspose3d(192,128,kernel_size=2,stride=2)
        #16*16*16
        #
        self.decoder_layer3=nn.Sequential(
            nn.Conv3d(128*2,128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(128),
            )
        #self.resup3=ResBlock3D(128*2, 128)
        self.decoder_layer3_up=nn.ConvTranspose3d(128,64,kernel_size=2,stride=2)
        #32*32*32
        #
        self.decoder_layer4=nn.Sequential(
            nn.Conv3d(64*2,64,kernel_size=3,stride=1,padding=1),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2,padding=1),
            nn.Conv3d(32,1,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2)
            )
        #64*64*64
    def forward(self,x,mri):
        #
        mri0=self.encoder_layer0_mri(mri)
        mri1 = self.encoder_layer1_mri(mri0)
        mri2 = self.encoder_layer2_mri(mri1)
        #
        #x=self.encoder_conv(x)
        en0=self.fuse0(self.encoder_layer0_down(x),mri0)#32*32*32
        #print("en0:",en0.shape)
        en1=self.fuse1(self.encoder_layer1_down(en0),mri1)#16*16*16
        #print("en1:",en1.shape)
        #
        en2=self.fuse2(self.encoder_layer2_down(en1),mri2)#8*8*8
        #print("en2:",en2.shape)
        en3=self.encoder_layer3_down(en2)#4*4*4
        #print("en3:",en3.shape)
        #
        de0=self.decoder_layer1(en3)
        de0=self.decoder_layer1_up(de0)#8*8*8
        #print("d0:",de0.shape)
        #
        cat1=torch.cat([en2,de0],1)
        de1=self.decoder_layer2(cat1)
        de1=self.decoder_layer2_up(de1)
        #print("d1:",de1.shape)
        del de0
        #
        cat2=torch.cat([en1,de1],1)
        de2=self.decoder_layer3(cat2)
        de2=self.decoder_layer3_up(de2)
        del de1
        #print("d2:",de2.shape)
        #
        cat3=torch.cat([en0,de2],1)
        de3=self.decoder_layer4(cat3)
        del de2,en0,en1,en2,en3
        #print("d3:",de3.shape)
        return de3+x
#
#
if __name__ == '__main__':
    #emb=PatchEmbed(img_size=64,patch_size=[3,5,7],in_chans=1,embed_dim=64)
    #emb2=PatchEmbed(img_size=32,patch_size=[3,5,7],in_chans=64,embed_dim=128)
    model=Generator()
    #total_params = sum(p.numel() for p in model.parameters())
    #print(total_params)
    x=torch.randn(1,1,64,64,64)
    y=torch.randn(1,1,64,64,64)
    #
    #print(emb(x).shape)
    #print(emb2(emb(x)).shape)
    #
    res=model(x,y)
    print(res.shape)
    #summary(model,[(64,16,16,16),(64,16,16,16)],device='cuda')