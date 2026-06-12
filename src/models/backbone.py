import os
import torch
import timm

def _strip_prefix(state_dict, prefix='model.'):
    """Strips the model prefix from checkpoint state dict keys."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_sd[new_key] = v
    return new_sd


def load_retfound_green(weights_path: str, img_size: int = 224):
    """
    Load RETFound-Green backbone — ViT-Base (768-d features).
    Uses vit_base_patch16_224 instead of vit_large_patch16_224.
    """
    backbone = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        img_size=img_size,
        num_classes=0,
        global_pool='avg',
    )

    if not os.path.exists(weights_path):
        print(f'WARNING: Weights not found at {weights_path}. Using random initialization.')
        return backbone

    ckpt = torch.load(weights_path, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    state_dict = _strip_prefix(state_dict, 'model.')
    
    # Keep only encoder keys (drop decoder, mask_token, norm_pix_loss, etc.)
    encoder_keys = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(p) for p in ['decoder', 'mask_token', 'norm_pix_loss'])
    }

    missing, unexpected = backbone.load_state_dict(encoder_keys, strict=False)
    print(f'RETFound-Green weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}')
    if missing:
        print(f'  Sample missing keys: {missing[:5]}')
    return backbone
