"""
TwinLiteNet-MFA: Multi-Scale Feature Aggregation with Channel-Spatial AoA Attention
Model Architecture

This file contains the core model architecture including:
- ModelConfig: Architecture configurations
- ChannelSpatialAoA: Channel and Spatial Attention Module
- MultiScaleSEBlock: Multi-scale block with SE attention
- CAAM: Context-Aware Attention Module
- Encoder: Feature extraction backbone
- TwinLiteNetPlus: Main model for dual-task segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# ATTENTION MODULES
# ============================================================================

class ChannelSpatialAoA(nn.Module):
    """Channel and Spatial Attention Module (AoA)"""
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
        # Channel attention
        z = F.adaptive_avg_pool2d(x, 1).view(B, C)
        w_c = self.sigmoid(self.fc2(self.relu(self.fc1(z)))).view(B, C, 1, 1)
        x_chan_att = x * w_c
        
        # Spatial attention
        avg_map = torch.mean(x_chan_att, dim=1, keepdim=True)
        max_map, _ = torch.max(x_chan_att, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_map, max_map], dim=1)
        w_s = self.sigmoid(self.spatial_conv(spatial_input))
        x_spatial_att = x * w_s
        
        # Refinement
        z_refine = F.adaptive_avg_pool2d(x_spatial_att, 1).view(B, C)
        w_c_refine = self.sigmoid(self.fc2_refine(self.relu(self.fc1_refine(z_refine)))).view(B, C, 1, 1)
        att_map = self.sigmoid(w_c_refine + w_s)
        
        return x * att_map


class MultiScaleSEBlock(nn.Module):
    """Multi-scale block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=8, dropout_rate=0.0):
        super(MultiScaleSEBlock, self).__init__()
        bottleneck_channels = max(1, in_channels // 4)
        
        # Branch 0: 1x1 convolution
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 1: 1x1 + 3x3 depthwise convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 + 5x5 depthwise convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=5, 
                     stride=stride, padding=2, groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion and attention
        total_branch_channels = bottleneck_channels * 3
        self.conv_fuse = nn.Conv2d(total_branch_channels, out_channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(out_channels)
        self.aoa_attention = ChannelSpatialAoA(channels=out_channels, reduction=reduction_ratio)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        
        # Skip connection
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


# ============================================================================
# CONTEXT-AWARE ATTENTION MODULE (CAAM)
# ============================================================================

def patch_split(input, bin_size):
    """Split feature map into patches"""
    B, C, H, W = input.size()
    bin_num_h, bin_num_w = bin_size[0], bin_size[1]
    rH, rW = H // bin_num_h, W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()
    out = out.view(B, -1, rH, rW, C)
    return out


def patch_recover(input, bin_size):
    """Recover feature map from patches"""
    B, N, rH, rW, C = input.size()
    bin_num_h, bin_num_w = bin_size[0], bin_size[1]
    H, W = rH * bin_num_h, rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GCN(nn.Module):
    """Graph Convolutional Network"""
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


# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

class ConvBatchnormRelu(nn.Module):
    """Conv + BatchNorm + PReLU block"""
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
    """Depthwise Separable Convolution"""
    def __init__(self, nIn, nOut, kSize, stride, d):
        super().__init__()
        padding = d * (kSize - 1) // 2
        self.depthwise = nn.Conv2d(nIn, nIn, kSize, stride=stride, padding=padding, 
                                   dilation=d, groups=nIn, bias=False)
        self.pointwise = nn.Conv2d(nIn, nOut, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BatchnormRelu(nn.Module):
    """BatchNorm + PReLU block"""
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class DepthwiseESP(nn.Module):
    """Depthwise ESP (Efficient Spatial Pyramid) block"""
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
            residual = self.proj(input) if self.proj is not None else input
            if combine.shape[2:] != residual.shape[2:]:
                combine = F.interpolate(combine, size=residual.shape[2:], mode='bilinear', align_corners=False)
            combine = residual + combine
        
        output = self.bn(combine)
        return output


class AvgDownsampler(nn.Module):
    """Average Pooling Downsampler"""
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
    """Simple Upsampling Block"""
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
    """Upsampling + Convolution Block"""
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


# ============================================================================
# ENCODER
# ============================================================================

class Encoder(nn.Module):
    """Feature Extraction Encoder"""
    def __init__(self, config_name):
        super().__init__()
        chanel_img = ModelConfig.chanel_img
        model_cfg = ModelConfig.sc_ch_dict[config_name]
        
        # Level 1
        self.level1 = ConvBatchnormRelu(chanel_img, model_cfg['chanels'][0], stride=2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)
        self.b1 = ConvBatchnormRelu(model_cfg['chanels'][0] + chanel_img, model_cfg['chanels'][1])
        
        # Level 2
        self.level2_0 = MultiScaleSEBlock(model_cfg['chanels'][1], model_cfg['chanels'][2], 
                                           stride=2, reduction_ratio=8, dropout_rate=0.1)
        self.level2 = nn.ModuleList()
        for i in range(model_cfg['p']):
            self.level2.append(DepthwiseESP(model_cfg['chanels'][2], model_cfg['chanels'][2]))
        
        # Level 3
        self.b2 = ConvBatchnormRelu(model_cfg['chanels'][2] * 2 + chanel_img, model_cfg['chanels'][3])
        self.level3_0 = MultiScaleSEBlock(model_cfg['chanels'][3], model_cfg['chanels'][3], 
                                           stride=2, reduction_ratio=8, dropout_rate=0.1)
        self.level3 = nn.ModuleList()
        for i in range(model_cfg['q']):
            self.level3.append(DepthwiseESP(model_cfg['chanels'][3], model_cfg['chanels'][3]))
        
        self.b3 = ConvBatchnormRelu(model_cfg['chanels'][3] * 2, model_cfg['chanels'][2])

    def forward(self, input):
        # Level 1
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        
        # Level 2
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        
        # Level 3
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


# ============================================================================
# MAIN MODEL
# ============================================================================

class TwinLiteNetPlus(nn.Module):
    """
    TwinLiteNet-MFA: Multi-Scale Feature Aggregation with Channel-Spatial AoA Attention
    
    A lightweight dual-task segmentation model for:
    - Drivable area segmentation
    - Lane line detection
    
    Args:
        config_name: Model size - 'nano', 'small', 'medium', or 'large'
    """
    def __init__(self, config_name='nano'):
        super().__init__()
        chanel_img = ModelConfig.chanel_img
        model_cfg = ModelConfig.sc_ch_dict[config_name]
        
        # Encoder
        self.encoder = Encoder(config_name)
        
        # Context-Aware Attention Module
        self.caam = CAAM(feat_in=model_cfg['chanels'][2], 
                        num_classes=model_cfg['chanels'][2], 
                        bin_size=(2, 4), 
                        norm_layer=nn.BatchNorm2d)
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
        # Encoder
        out_encoder, inp1, inp2 = self.encoder(input)
        
        # Context-Aware Attention
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
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(config_name='nano'):
    """Get model information"""
    model = TwinLiteNetPlus(config_name)
    num_params = count_parameters(model)
    
    info = {
        'config': config_name,
        'parameters': num_params,
        'parameters_M': f"{num_params / 1e6:.2f}M",
        'architecture': ModelConfig.sc_ch_dict[config_name]
    }
    return info


if __name__ == '__main__':
    # Test model
    for config in ['nano', 'small', 'medium', 'large']:
        print(f"\n{config.upper()} Model:")
        info = get_model_info(config)
        print(f"  Parameters: {info['parameters_M']}")
        print(f"  Architecture: {info['architecture']}")
        
        # Test forward pass
        model = TwinLiteNetPlus(config)
        model.eval()
        x = torch.randn(1, 3, 384, 640)
        with torch.no_grad():
            out_da, out_ll = model(x)
            print(f"  Output shape - DA: {out_da.shape}, LL: {out_ll.shape}")
