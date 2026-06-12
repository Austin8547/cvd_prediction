import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss for binary classification with optional label smoothing and positive weighting.
    """
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.05, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        eps = self.label_smoothing
        targets_smooth = targets * (1 - eps) + (1 - targets) * eps

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_w = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_w * bce
        return loss.mean()
