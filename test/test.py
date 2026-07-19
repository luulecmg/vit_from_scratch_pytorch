import torch
import torch.nn as nn
import argparse
from modeling.vit import ViT, TransformerEncoder, MLP, MultiHeadAttention


torch.manual_seed(0)
image = torch.rand(2, 1, 28, 28)


def build_simple_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--in_channel", type=int, default=1)
    parser.add_argument("--embed_channel", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--mlp_scale", type=int, default=2)
    parser.add_argument("--activation", type=str, default="gelu")
    return parser.parse_args()


def test_multihead_attention():
    print("\n---Test MultiHeadAttention")
    attention_layer = MultiHeadAttention(embed_dim=64, num_head=4)
    x = torch.rand((2, 9, 64))
    output = attention_layer(x)
    print(f"Attention output shape: {output.shape}")


def test_transformer_encoder():
    print("\n---Test TransformerEncoder")
    transformer_encoder = TransformerEncoder(embed_dim=64, num_head=4, mlp_scale=2, activation=nn.GELU)
    x = torch.rand((2, 9, 64))
    output = transformer_encoder(x)
    print(f"Transformer output shape: {output.shape}")


def test_mlp():
    print("\n---Test MLP")
    mlp_layer = MLP(embed_dim=64, mlp_scale=2, activation=nn.GELU)
    x = torch.rand((2, 9, 64))
    output = mlp_layer(x)
    print(f"MLP output shape: {output.shape}")


def test_vit():
    print("\n---Test ViT")
    args = build_simple_args()
    vit_model = ViT(args)
    logits = vit_model(image)
    print(f"ViT output shape: {logits.shape}")


if __name__ == '__main__':
    test_multihead_attention()
    test_transformer_encoder()
    test_mlp()
    test_vit()
    print("\nAll tests passed!")