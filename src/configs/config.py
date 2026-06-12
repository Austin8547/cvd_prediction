import os
import torch

# ─── PATHS ───────────────────────────────────────────────────
JSON_PATH = '/home/dfsaustin/cv_work/data-pred/data_info.json'
IMAGE_FOLDER = '/home/dfsaustin/cv_work/data-pred/Fundus_CIMT_2903/Fundus_CIMT_2903_Dataset'
WEIGHTS_PATH = 'RETFound_oct_weights.pth'   # RETFound-Green weights
BEST_MODEL_PATH = 'best_cimt_green_multimodal_v2.pth'
AUG_SAVE_DIR = '/home/dfsaustin/cv_work/cvd_prediction/src/data/augmented'


# Ensure directories exist
os.makedirs(AUG_SAVE_DIR, exist_ok=True)

# ─── DEVICE ──────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── HYPERPARAMETERS ─────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 8        # ViT-Base is lighter so could go 16 if needed
EPOCHS = 120
SKIP_AUGMENTATION = False
NUM_WORKERS = 2

# ─── DIFFERENTIAL LEARNING RATES ─────────────────────────────
BACKBONE_LR = 5e-6
HEAD_LR = 5e-5
CLINICAL_LR = 5e-5

# ─── COSINE WARMUP SCHEDULER ─────────────────────────────────
WARMUP_EPOCHS = 8
ETA_MIN = 5e-7

# ─── CLINICAL BRANCH ─────────────────────────────────────────
CLINICAL_FEAT_DIM = 128

# ─── DATA AUGMENTATION ───────────────────────────────────────
AUG_MULTIPLIER = 5
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# ─── LOSS FUNCTION ───────────────────────────────────────────
LABEL_SMOOTHING = 0.05
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

# ─── GRADIENT CLIPPING ───────────────────────────────────────
GRAD_CLIP_NORM = 1.0

# ─── TEST-TIME AUGMENTATION (TTA) ────────────────────────────
TTA_N_PASSES = 5

# ─── CLASS DEFINITIONS ───────────────────────────────────────
CLASS_NAMES = ['CIMT Normal', 'CIMT Thickened']

# ─── EARLY STOPPING ──────────────────────────────────────────
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 5e-4

# ─── PROGRESSIVE UNFREEZING SCHEDULE ─────────────────────────
# Format: epoch_num: (start_block_index, end_block_index)
UNFREEZE_SCHEDULE = {
    30: (-8, -4),
    60: (-12, -8),
}
