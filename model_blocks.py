import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return F.relu(x)

# Channel attention module.
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial attention module.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        attention_weights = self.sigmoid(out)
        return attention_weights

# CBAM attention module.
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        self.save_attention = False
        self.spatial_weights = None
        self.channel_weights = None
    
    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        if self.save_attention:
            self.channel_weights = channel_att.detach()
            self.spatial_weights = spatial_att.detach()
        
        return x

# Residual block with CBAM attention.
class ResidualBlockCBAM(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(ResidualBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.cbam = CBAM(n)
    
    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.cbam(y)
        return y + x


# DEM feature extraction module.
class DEMFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEMFeatureExtractor, self).__init__()
        # Gradient feature extraction.
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize gradient operators.
        vertical_kernel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        horizontal_kernel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        diagonal_kernel = torch.FloatTensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            self.conv_v.weight.data.copy_(vertical_kernel.repeat(out_channels, in_channels, 1, 1))
            self.conv_h.weight.data.copy_(horizontal_kernel.repeat(out_channels, in_channels, 1, 1))
            self.conv_d.weight.data.copy_(diagonal_kernel.repeat(out_channels, in_channels, 1, 1))
        
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        grad_v = self.conv_v(x)
        grad_h = self.conv_h(x)
        grad_d = self.conv_d(x)
        
        grad_features = torch.cat([grad_v, grad_h, grad_d], dim=1)
        fused_features = self.fusion(grad_features)
        fused_features = self.activation(fused_features)
        enhanced_features = self.cbam(fused_features)
        
        return enhanced_features
