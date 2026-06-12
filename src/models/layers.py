import torch
import torch.nn as nn

class BilateralAttentionGate(nn.Module):
    """
    Lightweight attention gate for reweighting bilateral (right + left eye) features.
    Input : [B, feat_dim]
    Output: [B, feat_dim] (attention-weighted)
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 8),
            nn.GELU(),
            nn.Linear(feat_dim // 8, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ClinicalMLP(nn.Module):
    """
    Deep MLP to project raw clinical features (age & gender) into a high-dimensional feature space.
    Input : [B, in_dim]
    Output: [B, out_dim]
    """
    def __init__(self, in_dim: int = 3, out_dim: int = 128):
        super().__init__()
        self.clinical_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.clinical_mlp(x)
