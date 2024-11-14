import torch
from torch import nn
import torch.nn.functional as F

from model.DepthwiseSeperableConv import DepthwiseSeparableConv2d


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x) # (B, E, n, n) which B is batch size, E is embed dim, n is number of patches in row or column
        x = x.flatten(2) # (B, E, N)  where N equals number of patches
        x = x.transpose(1, 2) # (B, N, E)
        x = self.patch_norm(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    # Input shape: (B, N, E)
    # Output shape: (B, N, E)
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape # B is batch size, N is number of patches, C is embed dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) # Shape = (B, N, 3, num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).chunk(3, dim=0) # each Shape = (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) # Shape = (B, num_heads, N, N)
        attn = F.softmax(attn, -1) + 1e-8 # Shape = (B, num_heads, N, N)

        out = (attn @ v).transpose(-2, -1) # shape =
        out = out.reshape(B, N, C)
        out = self.fc_out(out)
        return out


class TransformerEncoderLayer(nn.Module):
    # Input shape : (B, N, E)
    # Output shape : (B, N, E)
    def  __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super(TransformerEncoderLayer, self).__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads) # Shape would be (B, N, E)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.attn_norm(x + self.attn(x))
        x = self.mlp_norm(x + self.mlp(x))
        x = self.relu(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, embed_dim, num_heads, mlp_ratio, depth):
        super(VisionTransformer, self).__init__()

        self.skip_connections = None
        self.patch_embed = PatchEmbedding(img_size=img_size, in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])


        self.out_attn = MultiHeadSelfAttention(embed_dim, num_heads=1)
        self.norm_out = nn.LayerNorm(embed_dim * self.patch_embed.num_patches)
        self.out_mlp = nn.Linear(embed_dim * self.patch_embed.num_patches, 4)

    def forward(self, x):
        x = self.patch_embed(x) # shape : (B, N, E), which B is batch size, N is number of patches, E is embed dim
        x = x + self.pos_embed

        for i, layer in enumerate(self.transformer_layers):
            x = x + layer(x)  # (B, N, E)

        x = self.out_attn(x) # Shape : (B, N, E)
        x = x.view(x.shape[0], -1) # Shape : (B, N*E)
        x = self.norm_out(x) # Shape : (B, N*E)
        x = self.out_mlp(x) # Shape : (B, 4)
        return x



