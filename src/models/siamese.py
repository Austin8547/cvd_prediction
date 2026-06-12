import torch
import torch.nn as nn
from models.backbone import load_retfound_green
from models.layers import BilateralAttentionGate, ClinicalMLP

class GreenMultimodalSiamese(nn.Module):
    """
    Multimodal Siamese network combining shared RETFound-Green ViT backbone,
    a bilateral attention gate, clinical feature MLP, and a late fusion classifier.
    """
    def __init__(self, weights_path: str, img_size: int = 224,
                 clinical_in_dim: int = 3, clinical_feat_dim: int = 128):
        super().__init__()

        # Shared image encoder (Siamese)
        self.backbone = load_retfound_green(weights_path, img_size)

        # Freeze all backbone layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last 4 Transformer blocks + final LayerNorm layer
        for param in self.backbone.blocks[-4:].parameters():
            param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        feat_dim = self.backbone.num_features  # 768 for ViT-Base

        # Bilateral attention gate (features combined: 2 * 768 = 1536)
        self.bilateral_attn = BilateralAttentionGate(feat_dim=2 * feat_dim)

        # Clinical feature MLP branch
        self.clinical_mlp = ClinicalMLP(in_dim=clinical_in_dim, out_dim=clinical_feat_dim)

        # Fusion classifier
        fusion_dim = (2 * feat_dim) + clinical_feat_dim  # 1536 + 128 = 1664

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

        print(f'GreenMultimodalSiamese initialised:')
        print(f'  Backbone           : RETFound-Green ViT-Base')
        print(f'  Image feat dim     : {feat_dim} per eye -> {2*feat_dim} bilateral')
        print(f'  Bilateral attn gate: enabled ({2*feat_dim}-d)')
        print(f'  Clinical feat dim  : {clinical_feat_dim} (3 -> 64 -> {clinical_feat_dim})')
        print(f'  Fusion input dim   : {fusion_dim}')
        print(f'  Unfrozen blocks    : last 4 + norm')

    def encode_image(self, x):
        """Helper to extract features from a single eye image tensor."""
        return self.backbone(x)

    def encode_clinical(self, clinical):
        """Helper to project clinical features through the MLP."""
        return self.clinical_mlp(clinical)

    def forward(self, right_eye, left_eye, clinical):
        # Extract features for both eyes
        f_r = self.encode_image(right_eye)     # [B, 768]
        f_l = self.encode_image(left_eye)      # [B, 768]
        
        # Concatenate bilateral features
        f_img = torch.cat([f_r, f_l], dim=1)   # [B, 1536]
        f_img = self.bilateral_attn(f_img)     # [B, 1536]

        # Process clinical features
        f_clin = self.encode_clinical(clinical) # [B, 128]

        # Fusion and classification
        fused = torch.cat([f_img, f_clin], dim=1) # [B, 1664]
        return self.classifier(fused).squeeze(1)   # [B]
