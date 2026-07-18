import torch
import torch.nn as nn
from modeling.vit import ViT, TransformerEncoder, MLP, MultiHeadAttention


torch.manual_seed(0)
image = torch.rand(2, 1, 28, 28)


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
    vit_model = ViT(
        image_size=28,
        in_channel=1,
        embed_chanel=64,
        patch_size=7,
        num_class=10,
        depth=2,
        num_head=4,
        mlp_scale=2,
        activation=nn.GELU,
    )
    logits = vit_model(image)
    print(f"ViT output shape: {logits.shape}")


if __name__ == '__main__':
    test_multihead_attention()
    test_transformer_encoder()
    test_mlp()
    test_vit()
    print("\nAll tests passed!")