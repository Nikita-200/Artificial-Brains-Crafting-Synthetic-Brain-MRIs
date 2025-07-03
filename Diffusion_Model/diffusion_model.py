import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Generates sinusoidal position embeddings for diffusion timesteps
    Adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    assert len(timesteps.shape) == 1  # (B,)
    
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    # Zero padding if embedding_dim is odd
    if embedding_dim % 2 == 1:  
        emb = F.pad(emb, (0, 1, 0, 0))
    
    return emb  # Returns shape (B, embedding_dim)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_emb_proj = nn.Linear(time_emb_dim, in_channels * 2)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), 
            # nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels) 
            # nn.InstanceNorm2d(out_channels)
        )
        
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        if t_emb is not None:
            scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
            x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            
        return F.gelu(self.block(x) + self.residual(x))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.shape
        attn = self.net(self.gap(x).view(b, c))
        return x * attn.view(b, c, 1, 1)


class SliceDiffLite(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, base_channels=128, time_emb_dim=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # Simplified Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        # === New conditional encoders ===
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, base_channels//4, 3, padding=1),
            nn.InstanceNorm2d(base_channels//4),
            nn.GELU()
        )
        self.blur_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels//4, 3, padding=1),
            nn.InstanceNorm2d(base_channels//4),
            nn.GELU()
        )
        
        # # Lightweight Condition Encoder
        # self.cond_encoder = nn.Sequential(
        #     nn.Conv2d(cond_channels, base_channels//2, 3, padding=1),
        #     nn.InstanceNorm2d(base_channels//2),
        #     nn.GELU()
        # )
        
        # Encoder with Reduced Capacity
        self.enc1 = ResidualBlock(in_channels + base_channels//2, base_channels, time_emb_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels*2, time_emb_dim)
        
        # Bottleneck with Single Attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels*2, base_channels*4, time_emb_dim),
            ChannelAttention(base_channels*4),
            ResidualBlock(base_channels*4, base_channels*4, time_emb_dim)
        )
        
        # Decoder with Skip Connections
        self.dec1 = ResidualBlock(base_channels*6, base_channels*2, time_emb_dim)
        self.dec2 = ResidualBlock(base_channels*3, base_channels, time_emb_dim)
        
        self.final = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, seg_mask, blurred_img, t):
    # def forward(self, x, cond, t):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Early Fusion of Condition
        # cond_feat = self.cond_encoder(cond)
        seg_feat = self.seg_encoder(seg_mask)
        blur_feat = self.blur_encoder(blurred_img)    
        cond_feat = torch.cat([seg_feat, blur_feat], dim=1)
        
        x = torch.cat([x, cond_feat], dim=1)
        
        # Encoder
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(F.max_pool2d(x1, 2), t_emb)
        
        # Bottleneck
        x3 = self.bottleneck(F.max_pool2d(x2, 2))
        
        # Decoder
        x = F.interpolate(x3, scale_factor=2)
        x = self.dec1(torch.cat([x, x2], 1), t_emb)
        x = F.interpolate(x, scale_factor=2)
        x = self.dec2(torch.cat([x, x1], 1), t_emb)
        
        return self.final(x)
