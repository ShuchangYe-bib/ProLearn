# === Standard Library ===
import math

# === Third-party Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.networks.blocks.unetr_block import UnetrUpBlock


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding used in Transformer models.
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create matrix of shape (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, L, d_model]
        Returns:
            Tensor: [B, L, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class SelfAttentionLayer(nn.Module):
    """
    Applies self-attention with positional encoding.
    """
    def __init__(self, in_channels: int, n_heads: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.vis_pos = PositionalEncoding(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=n_heads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.vis_pos(x)
        y = self.norm(x)
        y, _ = self.self_attn(y, y, y)
        return x + self.self_attn_norm(y)


class CrossAttentionLayer(nn.Module):
    """
    Implements cross-attention between image and text embeddings.
    """
    def __init__(self, in_channels: int, output_text_len: int, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=n_heads, batch_first=True)
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image = image + self.vis_pos(image)
        text = text + self.txt_pos(text)
        image_norm = self.norm(image)
        attn_output, _ = self.cross_attn(query=image_norm, key=text, value=text)
        return image + self.scale * self.cross_attn_norm(attn_output)


class GuideDecoderLayer(nn.Module):
    """
    Combines self-attention and cross-attention for multimodal fusion.
    """
    def __init__(self, in_channels: int, output_text_len: int, input_text_len: int = 24, embed_dim: int = 768):
        super().__init__()
        self.self_attn = SelfAttentionLayer(in_channels)
        self.cross_attn = CrossAttentionLayer(in_channels, output_text_len)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, N, C1] Visual features
            txt (Tensor): [B, L, C] Text features
        Returns:
            Tensor: Attended visual features
        """
        txt = self.text_project(txt)
        x = self.self_attn(x)
        x = self.cross_attn(x, txt)
        return x


class GuideDecoder(nn.Module):
    """
    Upsampling decoder block with attention-guided fusion.
    """
    def __init__(self, in_channels: int, out_channels: int, spatial_size: int,
                 output_text_len: int, input_text_len: int = 24, embed_dim: int = 768):
        super().__init__()
        self.guide_layer = GuideDecoderLayer(in_channels, output_text_len, input_text_len, embed_dim)
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='BATCH'
        )

    def forward(self, vis: torch.Tensor, skip_vis: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vis (Tensor): [B, N, C] visual features
            skip_vis (Tensor): [B, N', C] skip features
            txt (Tensor): [B, L, C] text features
        Returns:
            Tensor: [B, N', C_out]
        """
        if txt is not None:
            vis = self.guide_layer(vis, txt)

        vis = rearrange(vis, 'B (H W) C -> B C H W', H=self.spatial_size, W=self.spatial_size)
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W', H=self.spatial_size * 2, W=self.spatial_size * 2)

        out = self.decoder(vis, skip_vis)
        out = rearrange(out, 'B C H W -> B (H W) C')
        return out


class AttentionApproximation(nn.Module):
    """
    Approximates visual features using attention-weighted text features.
    """
    def __init__(self, text_dim: int, num_candidates: int, vis_dim: int):
        super().__init__()
        self.projection = nn.Linear(vis_dim, text_dim)
        self.self_attn = SelfAttentionLayer(text_dim)
        self.cross_attn = CrossAttentionLayer(text_dim, num_candidates)

    def forward(self, txt: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        img = self.projection(img)
        txt = self.self_attn(txt)
        txt = self.cross_attn(txt, img)
        return txt


class GuidedApproximation(nn.Module):
    """
    Refines prototype-text fusion via dual cross-attention and skip connections.
    """
    def __init__(self, text_dim: int, num_candidates: int, vis_dim: int, vis_skip_dim: int):
        super().__init__()
        self.projection1 = nn.Linear(vis_dim, text_dim)
        self.projection2 = nn.Linear(vis_skip_dim, text_dim)

        self.self_attn1 = SelfAttentionLayer(text_dim)
        self.self_attn2 = SelfAttentionLayer(text_dim)

        self.cross_attn1 = CrossAttentionLayer(text_dim, num_candidates)
        self.cross_attn2 = CrossAttentionLayer(text_dim, num_candidates)

        self.norm = nn.LayerNorm(text_dim)

    def forward(self, txt: torch.Tensor, vis: torch.Tensor, vis_skip: torch.Tensor) -> torch.Tensor:
        vis = self.projection1(vis)
        vis_skip = self.projection2(vis_skip)

        txt1 = self.cross_attn1(self.self_attn1(txt), vis)
        txt2 = self.cross_attn2(self.self_attn2(txt), vis_skip)

        return self.norm(txt + txt1 + txt2)
