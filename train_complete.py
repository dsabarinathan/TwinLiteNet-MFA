#!/usr/bin/env python
"""
TwinLiteNetPlus - Complete Training Script
All-in-one file with model, dataloader, loss, training, and validation

Version: 1.9
Date: 2025-01-24
Fixes:
  - v1.2: Windows path handling
  - v1.3: Config file respected
  - v1.4: Model architecture bugs fixed (DepthwiseESP residual connections)
  - v1.5: Fixed channel mismatch in Encoder b2 layer (67 ch -> 35 ch bug)
  - v1.6: Fixed label resize mismatch (360 -> 384 to match image height)
  - v1.7: Fixed validation shape mismatch (crop ground truth to match predictions)
  - v1.8: Added FLOPs calculation, parameters in M, auto-load best checkpoint
  - v1.9: Robust checkpoint loading, inspect utility, better error handling

Usage:
    python train_complete.py --config config.yaml
    python train_complete.py --inspect checkpoint.pth

Author: TwinLiteNetPlus Team
"""

__version__ = "1.9"

import os
import sys
import yaml
import math
import random
import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    import albumentations as A
    from skimage.filters import gaussian
    from skimage.restoration import denoise_bilateral
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install albumentations scikit-image Pillow")
    sys.exit(1)

# FLOPs calculation (optional)
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' not installed. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ModelConfig:
    """Model architecture configurations"""
    chanel_img = 3  # RGB images
    
    sc_ch_dict = {

         'nano': {
            'chanels': [8, 16, 24, 32],
            'p': 2,  # Number of ESP blocks in level 2
            'q': 3   # Number of ESP blocks in level 3
        },
        'small': {
            'chanels': [16, 24, 32, 64],
            'p': 2,  # Number of ESP blocks in level 2
            'q': 3   # Number of ESP blocks in level 3
        },
        'smallv2': {
            'chanels': [24, 32, 64, 128],
            'p': 3,
            'q': 5
        },
        'medium': {
            'chanels': [32, 48, 96, 192],
            'p': 5,
            'q': 8
        },
        'large': {
            'chanels': [48, 64, 128, 256],
            'p': 7,
            'q': 10
        }
    }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ChannelSpatialAoA(nn.Module):
    """Channel and Spatial Attention Module"""
    def __init__(self, channels, reduction=16, spatial_kernel=3):
        super(ChannelSpatialAoA, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.fc1_refine = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2_refine = nn.Linear(channels // reduction, channels, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, 
                                     padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        z = F.adaptive_avg_pool2d(x, 1).view(B, C)
        w_c = self.sigmoid(self.fc2(self.relu(self.fc1(z)))).view(B, C, 1, 1)
        x_chan_att = x * w_c
        avg_map = torch.mean(x_chan_att, dim=1, keepdim=True)
        max_map, _ = torch.max(x_chan_att, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_map, max_map], dim=1)
        w_s = self.sigmoid(self.spatial_conv(spatial_input))
        x_spatial_att = x * w_s
        z_refine = F.adaptive_avg_pool2d(x_spatial_att, 1).view(B, C)
        w_c_refine = self.sigmoid(self.fc2_refine(self.relu(self.fc1_refine(z_refine)))).view(B, C, 1, 1)
        att_map = self.sigmoid(w_c_refine + w_s)
        return x * att_map


class MultiScaleSEBlock(nn.Module):
    """Multi-scale block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=8, dropout_rate=0.0):
        super(MultiScaleSEBlock, self).__init__()
        bottleneck_channels = max(1, in_channels // 4)
        
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=5, 
                     stride=stride, padding=2, groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        total_branch_channels = bottleneck_channels * 3
        self.conv_fuse = nn.Conv2d(total_branch_channels, out_channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(out_channels)
        self.aoa_attention = ChannelSpatialAoA(channels=out_channels, reduction=reduction_ratio)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.skip_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out0 = self.branch0(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        combined = torch.cat([out0, out1, out2], dim=1)
        fused = self.conv_fuse(combined)
        fused = self.bn_fuse(fused)
        fused = self.aoa_attention(fused)
        fused = self.dropout(fused)
        out = fused + self.skip_proj(x)
        return F.relu(out, inplace=True)


def patch_split(input, bin_size):
    B, C, H, W = input.size()
    bin_num_h, bin_num_w = bin_size[0], bin_size[1]
    rH, rW = H // bin_num_h, W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()
    out = out.view(B, -1, rH, rW, C)
    return out


def patch_recover(input, bin_size):
    B, N, rH, rW, C = input.size()
    bin_num_h, bin_num_w = bin_size[0], bin_size[1]
    H, W = rH * bin_num_h, rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.PReLU(num_node)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class CAAM(nn.Module):
    """Context-Aware Attention Module"""
    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()
        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.PReLU(feat_in)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.PReLU(1)

    def forward(self, x):
        cam = self.conv_cam(x)
        cls_score = self.sigmoid(self.pool_cam(cam))
        residual = x
        cam = patch_split(cam, self.bin_size)
        x = patch_split(x, self.bin_size)
        B, _, rH, rW, K, C = cam.shape[0], cam.shape[1], cam.shape[2], cam.shape[3], cam.shape[-1], x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K)
        x = x.view(B, -1, rH*rW, C)
        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)
        pixel_confidence = F.softmax(cam, dim=2)
        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence
        local_feats = self.gcn(local_feats)
        global_feats = self.fuse(local_feats)
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1)
        query = self.proj_query(x)
        key = self.proj_key(local_feats)
        value = self.proj_value(global_feats)
        aff_map = torch.matmul(query, key.transpose(2, 3))
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)
        out = out.view(B, -1, rH, rW, value.shape[-1])
        out = patch_recover(out, self.bin_size)
        out_conv = self.conv_out(out)
        out = residual + out_conv
        return out


class ConvBatchnormRelu(nn.Module):
    def __init__(self, nIn, nOut, kSize=3, stride=1, groups=1, dropout_rate=0.0):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, d):
        super().__init__()
        # Calculate padding to maintain spatial dimensions
        # For dilation d and kernel k: padding = d * (k - 1) // 2
        padding = d * (kSize - 1) // 2
        self.depthwise = nn.Conv2d(nIn, nIn, kSize, stride=stride, padding=padding, 
                                   dilation=d, groups=nIn, bias=False)
        self.pointwise = nn.Conv2d(nIn, nOut, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseESP(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = max(int(nOut/5), 1)
        n1 = max(nOut - 4*n, 1)
        self.c1 = DepthwiseSeparableConv(nIn, n, 1, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1)
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2)
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4)
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8)
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16)
        self.bn = BatchnormRelu(nOut)
        self.add = add
        
        # Add projection if input/output channels don't match
        if add and nIn != nOut:
            self.proj = nn.Conv2d(nIn, nOut, 1, bias=False)
        else:
            self.proj = None

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        
        if self.add:
            # Project input if channel dimensions don't match
            residual = self.proj(input) if self.proj is not None else input
            
            # Handle spatial dimension mismatch due to dilated convolutions
            if combine.shape[2:] != residual.shape[2:]:
                # Resize combine to match residual spatial dimensions
                combine = F.interpolate(combine, size=residual.shape[2:], mode='bilinear', align_corners=False)
            
            combine = residual + combine
        
        output = self.bn(combine)
        return output


class BatchnormRelu(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class AvgDownsampler(nn.Module):
    def __init__(self, samplingTimes):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, 
                                         padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False):
        super(UpConvBlock, self).__init__()
        self.last = last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        if not last:
            self.conv = MultiScaleSEBlock(out_channels + sub_dim, out_channels, 
                                           stride=1, reduction_ratio=8, dropout_rate=0.1)
        else:
            self.conv = MultiScaleSEBlock(out_channels, out_channels, 
                                           stride=1, reduction_ratio=8, dropout_rate=0.1)

    def forward(self, x, ori_img=None):
        x = self.up_conv(x)
        if not self.last and ori_img is not None:
            x = torch.cat([x, ori_img], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config_name):
        super().__init__()
        chanel_img = ModelConfig.chanel_img
        model_cfg = ModelConfig.sc_ch_dict[config_name]
        
        self.level1 = ConvBatchnormRelu(chanel_img, model_cfg['chanels'][0], stride=2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)
        self.b1 = ConvBatchnormRelu(model_cfg['chanels'][0] + chanel_img, model_cfg['chanels'][1])
        self.level2_0 = MultiScaleSEBlock(model_cfg['chanels'][1], model_cfg['chanels'][2], 
                                           stride=2, reduction_ratio=8, dropout_rate=0.1)
        self.level2 = nn.ModuleList()
        for i in range(model_cfg['p']):
            self.level2.append(DepthwiseESP(model_cfg['chanels'][2], model_cfg['chanels'][2]))
        
        # Fix: b2 receives [output1, output1_0, inp2] = chanels[2]*2 + chanel_img
        self.b2 = ConvBatchnormRelu(model_cfg['chanels'][2] * 2 + chanel_img, model_cfg['chanels'][3])
        self.level3_0 = MultiScaleSEBlock(model_cfg['chanels'][3], model_cfg['chanels'][3], 
                                           stride=2, reduction_ratio=8, dropout_rate=0.1)
        self.level3 = nn.ModuleList()
        for i in range(model_cfg['q']):
            self.level3.append(DepthwiseESP(model_cfg['chanels'][3], model_cfg['chanels'][3]))
        
        self.b3 = ConvBatchnormRelu(model_cfg['chanels'][3] * 2, model_cfg['chanels'][2])

    def forward(self, input):
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        
        output2_cat = torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)
        return out_encoder, inp1, inp2


class TwinLiteNetPlus(nn.Module):
    """TwinLiteNetPlus model for dual-task segmentation"""
    def __init__(self, config_name='nano'):
        super().__init__()
        chanel_img = ModelConfig.chanel_img
        model_cfg = ModelConfig.sc_ch_dict[config_name]
        
        self.encoder = Encoder(config_name)
        self.caam = CAAM(feat_in=model_cfg['chanels'][2], num_classes=model_cfg['chanels'][2], 
                        bin_size=(2, 4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(model_cfg['chanels'][2], model_cfg['chanels'][1])
        
        # Drivable area decoder
        self.up_1_da = UpConvBlock(model_cfg['chanels'][1], model_cfg['chanels'][0], sub_dim=chanel_img)
        self.up_2_da = UpConvBlock(model_cfg['chanels'][0], 8, sub_dim=chanel_img)
        self.out_da = UpConvBlock(8, 2, last=True)
        
        # Lane line decoder
        self.up_1_ll = UpConvBlock(model_cfg['chanels'][1], model_cfg['chanels'][0], sub_dim=chanel_img)
        self.up_2_ll = UpConvBlock(model_cfg['chanels'][0], 8, sub_dim=chanel_img)
        self.out_ll = UpConvBlock(8, 2, last=True)

    def forward(self, input):
        out_encoder, inp1, inp2 = self.encoder(input)
        out_caam = self.caam(out_encoder)
        out_caam = self.conv_caam(out_caam)
        
        # Drivable area path
        out_da = self.up_1_da(out_caam, inp2)
        out_da = self.up_2_da(out_da, inp1)
        out_da = self.out_da(out_da)
        
        # Lane line path
        out_ll = self.up_1_ll(out_caam, inp2)
        out_ll = self.up_2_ll(out_ll, inp1)
        out_ll = self.out_ll(out_ll)
        
        return out_da, out_ll


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


class TotalLoss(nn.Module):
    """Combined loss for both tasks"""
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, outputs, targets):
        da_output, ll_output = outputs
        da_target, ll_target = targets
        
        if torch.cuda.is_available():
            da_target = da_target.cuda().float()
            ll_target = ll_target.cuda().float()
        
        focal_loss_da = self.focal_loss(da_output, da_target)
        tversky_loss_da = self.tversky_loss(da_output, da_target)
        focal_loss_ll = self.focal_loss(ll_output, ll_target)
        tversky_loss_ll = self.tversky_loss(ll_output, ll_target)
        
        focal_loss = focal_loss_da + focal_loss_ll
        tversky_loss = tversky_loss_da + tversky_loss_ll
        total_loss = focal_loss + tversky_loss
        
        return focal_loss, tversky_loss, total_loss


# ============================================================================
# DATASET AND AUGMENTATION
# ============================================================================

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image"""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """HSV color augmentation"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_perspective(combination, degrees=10, translate=0.1, scale=0.1, shear=10):
    """Apply random perspective transformation"""
    img, drivable, line = combination
    height, width = img.shape[0], img.shape[1]
    
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2
    
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    M = T @ S @ R @ C
    
    img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    drivable = cv2.warpAffine(drivable, M[:2], dsize=(width, height), borderValue=0)
    line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)
    
    return (img, drivable, line)


class BDD100KDataset(Dataset):
    """BDD100K Dataset for dual-task segmentation"""
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid
        self.Tensor = transforms.ToTensor()
        
        # Load augmentation parameters
        aug_cfg = config.get('augmentation', {})
        self.degrees = aug_cfg.get('degrees', 10)
        self.translate = aug_cfg.get('translate', 0.1)
        self.scale = aug_cfg.get('scale', 0.1)
        self.shear = aug_cfg.get('shear', 10)
        self.hgain = aug_cfg.get('hgain', 0.015)
        self.sgain = aug_cfg.get('sgain', 0.7)
        self.vgain = aug_cfg.get('vgain', 0.4)
        self.prob_perspective = aug_cfg.get('prob_perspective', 0.5)
        self.prob_flip = aug_cfg.get('prob_flip', 0.5)
        self.prob_hsv = aug_cfg.get('prob_hsv', 0.5)
        
        # Dataset paths
        data_cfg = config.get('dataset', {})
        self.dataset_root = data_cfg.get('root', './bdd100k')  # Store the actual root
        self.split = 'val' if valid else 'train'
        
        # Image directory
        self.image_dir = os.path.join(self.dataset_root, 'images', self.split)
        
        print(f"Dataset root: {os.path.abspath(self.dataset_root)}")
        print(f"Image directory: {os.path.abspath(self.image_dir)}")
        
        self.names = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Loaded {'validation' if valid else 'training'} dataset: {len(self.names)} images")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        W_, H_ = 640, 384
        image_name = os.path.join(self.image_dir, self.names[idx])
        
        image = cv2.imread(image_name)
        if image is None:
            raise ValueError(f"Failed to load image: {image_name}")
        
        # Load labels - proper path handling for Windows/Linux
        # Get the base filename without extension
        base_name = os.path.splitext(self.names[idx])[0]
        
        da_label_path = 'D:/Research/Light_Weight_Segmentation/TwinLiteNetPlus-main/TwinLiteNetPlus-main/bdd100k/drivable_are_annotations/'+self.split+'/'+f'{base_name}.png'

       # ll_label_path = os.path.join(self.dataset_root, 'lane_line_annotations', self.split, f'{base_name}.png')
        
        ll_label_path  = 'D:/Research/Light_Weight_Segmentation/TwinLiteNetPlus-main/TwinLiteNetPlus-main/bdd100k/lane_line_annotations/'+self.split+'/'+f'{base_name}.png'

        label1 = cv2.imread(da_label_path, 0)
        label2 = cv2.imread(ll_label_path, 0)
        
        if label1 is None or label2 is None:
            raise ValueError(f"Failed to load labels for: {image_name}\nDA path: {da_label_path}\nLL path: {ll_label_path}")
        
        # Apply augmentation
        if not self.valid:
            if random.random() < self.prob_perspective:
                combination = (image, label1, label2)
                image, label1, label2 = random_perspective(
                    combination, self.degrees, self.translate, self.scale, self.shear
                )
            
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
        
        image = letterbox(image, (H_, W_))
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        
        # Create binary masks
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)
        
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
        
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        
        return image_name, torch.from_numpy(image), (seg_da, seg_ll)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class SegmentationMetric:
    """Metrics for segmentation evaluation"""
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU

    def lineAccuracy(self):
        tp = self.confusionMatrix[1, 1]
        fp = self.confusionMatrix[0, 1]
        fn = self.confusionMatrix[1, 0]
        tn = self.confusionMatrix[0, 0]
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        return (sensitivity + specificity) / 2

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class ModelEMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


def poly_lr_scheduler(optimizer, init_lr, curr_epoch, max_epochs, power=1.5):
    """Polynomial learning rate decay"""
    lr = init_lr * (1 - curr_epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_size=(1, 3, 384, 640), device='cuda'):
    """Calculate FLOPs and parameters using thop"""
    if not THOP_AVAILABLE:
        return None, None
    
    try:
        model_copy = deepcopy(model)
        model_copy.eval()
        
        if device == 'cuda' and torch.cuda.is_available():
            model_copy = model_copy.cuda()
            input_tensor = torch.randn(input_size).cuda()
        else:
            input_tensor = torch.randn(input_size)
        
        flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        
        del model_copy
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return flops, params
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None, None


def format_number_M(num):
    """Format number in millions (M)"""
    return f"{num / 1e6:.2f}M"


# ============================================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, config, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (_, images, targets) in enumerate(pbar):
        if torch.cuda.is_available():
            images = images.cuda().float() / 255.0
        
        optimizer.zero_grad()
        
        # Use mixed precision training
        use_amp = torch.cuda.is_available()
        with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            outputs = model(images)
            focal_loss, tversky_loss, loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'focal': f'{focal_loss.item():.4f}',
            'tversky': f'{tversky_loss.item():.4f}'
        })
    
    return total_loss / num_batches, ema


@torch.no_grad()
def validate(model, val_loader, config):
    """Validate the model"""
    model.eval()
    
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    
    pbar = tqdm(val_loader, desc='Validation')
    
    for _, images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda().float() / 255.0
        
        outputs = model(images)
        
        # Drivable area evaluation
        out_da = outputs[0]
        target_da = targets[0]
        _, da_predict = torch.max(out_da, 1)
        da_predict = da_predict[:, 12:-12]
        _, da_gt = torch.max(target_da, 1)
        da_gt = da_gt[:, 12:-12]  # Crop to match prediction shape
        
        DA.reset()
        DA.addBatch(da_predict.cpu().numpy().flatten(), da_gt.cpu().numpy().flatten())
        
        # Lane line evaluation
        out_ll = outputs[1]
        target_ll = targets[1]
        _, ll_predict = torch.max(out_ll, 1)
        ll_predict = ll_predict[:, 12:-12]
        _, ll_gt = torch.max(target_ll, 1)
        ll_gt = ll_gt[:, 12:-12]  # Crop to match prediction shape
        
        LL.reset()
        LL.addBatch(ll_predict.cpu().numpy().flatten(), ll_gt.cpu().numpy().flatten())
    
    da_acc = DA.pixelAccuracy()
    da_iou = DA.IntersectionOverUnion()
    da_miou = DA.meanIntersectionOverUnion()
    
    ll_acc = LL.lineAccuracy()
    ll_iou = LL.IntersectionOverUnion()
    ll_miou = LL.meanIntersectionOverUnion()
    
    return (da_acc, da_iou, da_miou), (ll_acc, ll_iou, ll_miou)


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint structure and contents"""
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nInspecting checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dictionary with {len(checkpoint)} keys:")
            print("\nKeys found:")
            for key in checkpoint.keys():
                if key == 'state_dict':
                    state_dict = checkpoint[key]
                    print(f"  - {key}: dict with {len(state_dict)} parameters")
                    print(f"    First 5 keys: {list(state_dict.keys())[:5]}")
                elif key == 'optimizer':
                    print(f"  - {key}: optimizer state")
                elif isinstance(checkpoint[key], (int, float, str)):
                    print(f"  - {key}: {checkpoint[key]}")
                else:
                    print(f"  - {key}: {type(checkpoint[key])}")
            
            # Try to infer format
            if 'state_dict' in checkpoint:
                print("\n✓ Format: Full checkpoint (state_dict + optimizer + metadata)")
            elif any(k.startswith('encoder.') or k.startswith('caam.') for k in checkpoint.keys()):
                print("\n✓ Format: Direct state dict (model weights only)")
            else:
                print("\n⚠ Format: Unknown - may need custom loading")
        else:
            print(f"Checkpoint is not a dictionary: {type(checkpoint)}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(config):
    """Main training function"""
    
    # Setup
    print("=" * 80)
    print(f"TwinLiteNetPlus Training - Version {__version__}")
    print("=" * 80)
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # CUDA setup
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Using device: {device}")
    
    if cuda_available:
        cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create save directory
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Model
    model_config = config['model']['architecture']
    print(f"\nInitializing model: {model_config}")
    model = TwinLiteNetPlus(model_config)
    
    if cuda_available:
        model = model.cuda()
    
    # Display model statistics
    num_params = count_parameters(model)
    print(f"Total parameters: {num_params:,} ({format_number_M(num_params)})")
    
    # Calculate FLOPs
    if THOP_AVAILABLE:
        print("Calculating FLOPs...")
        flops, params_thop = calculate_flops(model, device=device)
        if flops is not None:
            print(f"FLOPs: {flops}")
            print(f"Parameters (thop): {params_thop}")
    
    # Auto-load best checkpoint if exists
    best_checkpoint_path = os.path.join(save_dir, 'checkpoint_best.pth')
    resume_path = config['training'].get('resume', '')
    auto_resume = config['training'].get('auto_resume', False)
    
    if not resume_path and os.path.isfile(best_checkpoint_path):
        if auto_resume:
            resume_path = best_checkpoint_path
            print(f"\n{'='*80}")
            print(f"Auto-loading best checkpoint: {best_checkpoint_path}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Found existing best checkpoint: {best_checkpoint_path}")
            try:
                load_best = input("Load best checkpoint? (y/n): ").lower().strip()
                if load_best == 'y':
                    resume_path = best_checkpoint_path
                    print(f"Loading best checkpoint...")
            except:
                # Handle non-interactive environments
                print("Running in non-interactive mode. Set 'auto_resume: true' in config to auto-load.")
            print(f"{'='*80}\n")
    
    # Data loaders
    print("\nLoading datasets...")
    train_dataset = BDD100KDataset(config, valid=False)
    val_dataset = BDD100KDataset(config, valid=True)
    
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Loss and optimizer
    criterion = TotalLoss()
    
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0001)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    # EMA
    use_ema = config['training'].get('use_ema', False)
    ema = ModelEMA(model) if use_ema else None
    if use_ema:
        print("Using Exponential Moving Average (EMA)")
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    max_epochs = config['training']['max_epochs']
    start_epoch = 0
    
    # Resume from checkpoint if specified
    resume_path = config['training'].get('resume', '')
    if resume_path and os.path.isfile(resume_path):
        print(f"\nResuming from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Format 1: Full checkpoint with state_dict
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    if 'optimizer' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch']
                    if use_ema and 'ema_state_dict' in checkpoint:
                        ema.ema.load_state_dict(checkpoint['ema_state_dict'])
                        ema.updates = checkpoint.get('updates', 0)
                    print(f"Resumed from epoch {start_epoch}")
                
                # Format 2: Direct state dict
                elif 'encoder.level1.conv.weight' in checkpoint or any(k.startswith('encoder.') for k in checkpoint.keys()):
                    model.load_state_dict(checkpoint)
                    print(f"Loaded model weights only")
                
                # Format 3: Unknown format
                else:
                    print(f"Checkpoint keys: {list(checkpoint.keys())}")
                    raise ValueError(f"Unrecognized checkpoint format. Keys found: {list(checkpoint.keys())[:5]}")
            else:
                raise ValueError(f"Checkpoint is not a dictionary, got type: {type(checkpoint)}")
            
            # Move model back to CUDA after loading from CPU
            if cuda_available:
                model = model.cuda()
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Available keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
            try:
                response = input("Continue without loading checkpoint? (y/n): ").lower().strip()
                if response != 'y':
                    sys.exit(1)
                else:
                    print("Continuing with fresh initialization...")
            except:
                print("Non-interactive mode: Exiting due to checkpoint error.")
                sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    
    best_miou = 0.0
    
    for epoch in range(start_epoch, max_epochs):
        # Update learning rate
        current_lr = poly_lr_scheduler(optimizer, lr, epoch, max_epochs)
        print(f"\nEpoch [{epoch+1}/{max_epochs}] - LR: {current_lr:.6f}")
        
        # Train
        train_loss, ema = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, ema
        )
        
        # Validate
        val_model = ema.ema if use_ema else model
        da_metrics, ll_metrics = validate(val_model, val_loader, config)
        
        print(f"\nResults:")
        print(f"  Drivable Area - Acc: {da_metrics[0]:.4f}, IoU: {da_metrics[1]:.4f}, mIoU: {da_metrics[2]:.4f}")
        print(f"  Lane Line     - Acc: {ll_metrics[0]:.4f}, IoU: {ll_metrics[1]:.4f}, mIoU: {ll_metrics[2]:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': current_lr,
            'da_miou': da_metrics[2],
            'll_miou': ll_metrics[2]
        }
        
        if use_ema:
            checkpoint['ema_state_dict'] = ema.ema.state_dict()
            checkpoint['updates'] = ema.updates
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        current_miou = (da_metrics[2] + ll_metrics[2]) / 2
        if current_miou > best_miou:
            best_miou = current_miou
            best_path = os.path.join(save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved! (mIoU: {best_miou:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training'].get('save_interval', 10) == 0:
            epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
        
        # Save model weights only
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        if use_ema:
            torch.save(ema.ema.state_dict(), model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best mIoU: {best_miou:.4f}")
    print("=" * 80)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config():
    """Create default configuration"""
    # Detect OS and adjust workers
    import platform
    default_workers = 2 if platform.system() == 'Windows' else 4
    
    return {
        'model': {
            'architecture': 'nano'  # nano, small, medium, large
        },
        'dataset': {
            'root': './bdd100k',
        },
        'augmentation': {
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.1,
            'shear': 10,
            'hgain': 0.015,
            'sgain': 0.7,
            'vgain': 0.4,
            'prob_perspective': 0.5,
            'prob_flip': 0.5,
            'prob_hsv': 0.5
        },
        'training': {
            'max_epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'num_workers': default_workers,
            'use_ema': True,
            'save_dir': './checkpoints',
            'save_interval': 10,
            'resume': '',
            'auto_resume': False  # Set to True to automatically load best checkpoint
        },
        'seed': 42
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwinLiteNetPlus Training')
    parser.add_argument('--version', action='store_true',
                       help='Show version and exit')
    parser.add_argument('--config', type=str, default='',
                       help='Path to config YAML file')
    parser.add_argument('--inspect', type=str, default='',
                       help='Inspect checkpoint file structure and exit')
    parser.add_argument('--model', type=str, default='',
                       choices=['', 'nano', 'small', 'medium', 'large'],
                       help='Model architecture (overrides config)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=0,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=0,
                       help='Learning rate (overrides config)')
    parser.add_argument('--data-root', type=str, default='',
                       help='Dataset root directory (overrides config file)')
    parser.add_argument('--save-dir', type=str, default='',
                       help='Directory to save checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume from checkpoint')
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable EMA')
    
    args = parser.parse_args()
    
    # Check for inspect mode
    if args.inspect:
        inspect_checkpoint(args.inspect)
        sys.exit(0)
    
    # Check version
    if args.version:
        print(f"TwinLiteNetPlus Training Script")
        print(f"Version: {__version__}")
        print(f"Date: 2025-01-24")
        print(f"\nChangelog:")
        print(f"  v1.2: Fixed Windows path handling")
        print(f"  v1.3: Fixed config file being ignored")
        print(f"  v1.4: Fixed model architecture bugs (DepthwiseESP)")
        print(f"  v1.5: Fixed channel mismatch in Encoder (b2 layer)")
        sys.exit(0)
    
    # Load or create config
    if args.config and os.path.isfile(args.config):
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override with command line arguments (only if explicitly provided)
    if args.model:
        config['model']['architecture'] = args.model
    if args.batch_size > 0:
        config['training']['batch_size'] = args.batch_size
    if args.epochs > 0:
        config['training']['max_epochs'] = args.epochs
    if args.lr > 0:
        config['training']['learning_rate'] = args.lr
    if args.data_root:
        config['dataset']['root'] = args.data_root
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir
    if args.resume:
        config['training']['resume'] = args.resume
    if args.no_ema:
        config['training']['use_ema'] = False
    
    # Print configuration
    print("\nConfiguration:")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 80 + "\n")
    
    # Start training
    main(config)
