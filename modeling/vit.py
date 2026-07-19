import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

activation_map = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
}

class ViT(nn.Module):
    '''
    Input:
        raw image
            (B, 1, 28, 28) for MNIST dataset
    
    Output:
        cls_token of n classes
            (B, 10) for MNIST dataset
    '''
    def __init__(self, args):

        super().__init__()
        self.embed_dim = args.embed_channel
        self.num_token = (args.image_size//args.patch_size) ** 2
        self.image_patchify = nn.Conv2d(args.in_channel, args.embed_channel, kernel_size=args.patch_size, stride=args.patch_size) # (B, 768, 16, 16)
        self.cls_token = nn.Parameter(torch.rand(1, args.embed_channel))
        self.pos_embed = nn.Parameter(torch.rand((self.num_token+1), args.embed_channel)) # (65, 768)

        self.transformers = nn.ModuleList([
            TransformerEncoder(embed_dim=args.embed_channel, num_head=args.num_head, mlp_scale=args.mlp_scale, activation=activation_map.get(args.activation, nn.GELU))
            for _ in range(args.depth)
        ])
        self.mlp_head = nn.Linear(args.embed_channel, args.num_class)
        self.model_name = f"ViT_patch{args.patch_size}_dim{args.embed_channel}_depth{args.depth}"
    
    def forward(self, image: Tensor):
        '''
        Args:
            original image of (B, C, H, W)

        Return:
            cls_token of (B, C)
        '''
        B, C, H, W = image.shape
        image_patches = self.image_patchify(image).permute(0, 2, 3, 1).reshape(B, self.num_token, self.embed_dim) # (B, 768, 16, 16) -> (B, 16**2, 768)
        cls_token = self.cls_token.expand(B, 1, -1)

        image_patches = torch.cat([cls_token, image_patches], dim=1) # (B, 65, 768)
        image_patches += self.pos_embed.expand(B, -1, self.embed_dim)

        for layer in self.transformers:
            image_patches = layer(image_patches)
        
        cls_token = image_patches[:, 0, :] # (B, 768)
        cls_token = self.mlp_head(cls_token) # (B, 10)
        return cls_token

class TransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 num_head: int = 12,
                 mlp_scale: int = 4,
                 activation: type[nn.Module] = nn.GELU,


    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attn = MultiHeadAttention(embed_dim=embed_dim, num_head=num_head)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_scale=mlp_scale, activation=activation)

    def forward(self, image_patches: Tensor):
        out = self.multihead_attn(self.norm1(image_patches))
        image_patches = image_patches + out

        out = self.mlp(self.norm2(image_patches))
        image_patches = image_patches + out

        return image_patches

class MLP(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 mlp_scale: int = 4,
                 activation: type[nn.Module] = nn.GELU,

    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_scale),
            activation(),
            nn.Linear(embed_dim * mlp_scale, embed_dim)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 num_head: int = 12,

    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3) # (768, 768*3)
        self.scale = (embed_dim//num_head)**-0.5
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor):
        '''
        Args:
            x: (B, N, 768)
        
        Returns:
            x: (B, N, 768) after multihead attention
        '''
        B, N, _ = x.shape
        qkv = self.qkv_proj(x) # (B, N, 768*3)
        qkv = qkv.reshape(B, N, self.num_head, -1, 3).permute(2, 0, 1, 3, 4) # (B, N, 12, 64, 3)

        q = qkv[..., 0] # (12, B, N, 64)
        k = qkv[..., 1]
        v = qkv[..., 2]

        attn = (self.scale * q @ k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).permute(1, 2, 0, 3).reshape(B, N, -1) # (B, N, 768)
        out = self.out_proj(out)
        return out



