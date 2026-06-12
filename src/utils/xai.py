import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from configs import config

# ViT-Base has patch size 16. With 224x224 input, spatial size is 14x14 patches.
PATCH_SIZE = 16
GRID_H = GRID_W = config.IMG_SIZE // PATCH_SIZE

def vit_reshape_transform(tensor):
    """
    Reshapes standard 1D ViT patch sequence outputs (dropping CLS token)
    into a 2D spatial feature map for spatial-based CAM computation.
    """
    spatial = tensor[:, 1:, :]  # drop CLS token
    B, N, C = spatial.shape
    spatial = spatial.reshape(B, GRID_H, GRID_W, C)
    return spatial.permute(0, 3, 1, 2).contiguous()


class RightEyeWrapper(nn.Module):
    """
    Wraps the multimodal model to compute GradCAM++ heatmaps for the right eye
    by freezing the left eye features and clinical features as static context.
    """
    def __init__(self, mm_model, fixed_left_feat, fixed_clin_feat):
        super().__init__()
        self.model = mm_model
        self.fixed_left_feat = fixed_left_feat
        self.fixed_clin_feat = fixed_clin_feat

    def forward(self, right_eye):
        f_r = self.model.encode_image(right_eye)
        f_img = torch.cat([f_r, self.fixed_left_feat.expand(f_r.size(0), -1)], dim=1)
        f_img = self.model.bilateral_attn(f_img)
        f_clin = self.fixed_clin_feat.expand(f_r.size(0), -1)
        fused = torch.cat([f_img, f_clin], dim=1)
        return self.model.classifier(fused)


class LeftEyeWrapper(nn.Module):
    """
    Wraps the multimodal model to compute GradCAM++ heatmaps for the left eye
    by freezing the right eye features and clinical features as static context.
    """
    def __init__(self, mm_model, fixed_right_feat, fixed_clin_feat):
        super().__init__()
        self.model = mm_model
        self.fixed_right_feat = fixed_right_feat
        self.fixed_clin_feat = fixed_clin_feat

    def forward(self, left_eye):
        f_l = self.model.encode_image(left_eye)
        f_img = torch.cat([self.fixed_right_feat.expand(f_l.size(0), -1), f_l], dim=1)
        f_img = self.model.bilateral_attn(f_img)
        f_clin = self.fixed_clin_feat.expand(f_l.size(0), -1)
        fused = torch.cat([f_img, f_clin], dim=1)
        return self.model.classifier(fused)


def compute_gradcam(wrapper_model, input_tensor, target_layer):
    """
    Computes the GradCAM++ heatmap overlay for a target image input.
    """
    target = [BinaryClassifierOutputTarget(0)]
    cam = GradCAMPlusPlus(
        model=wrapper_model,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform,
    )
    
    # Calculate heatmap map
    grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
    
    # Image normalization setup for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    rgb_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img * std + mean).clip(0, 1).astype(np.float32)
    
    # Generate heatmap overlay on the original image
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return overlay, grayscale_cam
