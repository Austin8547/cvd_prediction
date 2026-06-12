import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# Add current file's directory (src/) to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from configs import config
from data import get_dataloaders, run_offline_augmentation
from models import GreenMultimodalSiamese
from engine import (
    FocalBCEWithLogitsLoss, get_warmup_cosine_scheduler,
    run_epoch, handle_progressive_unfreeze
)
from utils import plot_training_curves


def main():
    print(f"Using device: {config.DEVICE}")

    # 1. Parse JSON metadata
    print(f"Loading metadata from {config.JSON_PATH}...")
    with open(config.JSON_PATH, 'r') as f:
        data_info = json.load(f)

    records = []
    for patient_id, info in data_info.items():
        gender_raw = info.get('gender', 0)
        gender_onehot = [1.0 - float(gender_raw), float(gender_raw)]
        age_norm = float(info.get('age', 0.0))

        records.append({
            'patient_id': patient_id,
            'right_eye': info['right_eye'],
            'left_eye': info['left_eye'],
            'label': int(info['label']),
            'group': int(info['group']),
            'gender_raw': gender_raw,
            'gender_female': gender_onehot[0],
            'gender_male': gender_onehot[1],
            'age_norm': age_norm,
            'true_age': info.get('True_age', None),
            'thickness': info.get('thickness', None),
        })

    df = pd.DataFrame(records)

    # Split original training, validation, and test data
    train_df = df[df['group'] == 1].reset_index(drop=True)
    val_df = df[df['group'] == 2].reset_index(drop=True)
    test_df = df[df['group'] == 3].reset_index(drop=True)

    print(f"Original Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # 2. Compute pos_weight on RAW Training Set (for BCE loss scaling)
    n_normal_raw = (train_df['label'] == 0).sum()
    n_thickened_raw = (train_df['label'] == 1).sum()
    softened_pw = float(n_normal_raw / n_thickened_raw) ** 0.5
    pos_weight_val = torch.tensor([softened_pw], dtype=torch.float32).to(config.DEVICE)
    print(f"BCEWithLogitsLoss pos_weight (softened raw): {pos_weight_val.item():.4f}")

    # 3. Run Offline Augmentation to balance classes
    augmented_train_df = run_offline_augmentation(
        train_df=train_df,
        image_folder=config.IMAGE_FOLDER,
        save_dir=config.AUG_SAVE_DIR,
        img_size=config.IMG_SIZE,
        force_rerun=config.SKIP_AUGMENTATION
    )

    # 4. Construct Dataloaders
    train_loader, val_loader, _, _ = get_dataloaders(
        train_df=augmented_train_df,
        val_df=val_df,
        test_df=test_df,
        image_folder=config.IMAGE_FOLDER,
        aug_folder=config.AUG_SAVE_DIR
    )

    # 5. Initialize Multimodal Siamese Model
    model = GreenMultimodalSiamese(
        weights_path=config.WEIGHTS_PATH,
        img_size=config.IMG_SIZE,
        clinical_in_dim=3,
        clinical_feat_dim=config.CLINICAL_FEAT_DIM
    ).to(config.DEVICE)

    # 6. Loss and Optimization Configuration
    if config.USE_FOCAL_LOSS:
        criterion = FocalBCEWithLogitsLoss(
            gamma=config.FOCAL_GAMMA,
            alpha=config.FOCAL_ALPHA,
            label_smoothing=config.LABEL_SMOOTHING,
            pos_weight=pos_weight_val
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

    # Group parameters for differential learning rates
    backbone_params = (
        list(model.backbone.blocks[-4:].parameters()) +
        list(model.backbone.norm.parameters())
    )
    bilateral_params = list(model.bilateral_attn.parameters())
    clinical_params = list(model.clinical_mlp.parameters())
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.BACKBONE_LR},
        {'params': bilateral_params, 'lr': config.HEAD_LR},
        {'params': clinical_params, 'lr': config.CLINICAL_LR},
        {'params': head_params, 'lr': config.HEAD_LR},
    ], weight_decay=5e-4)

    scheduler = get_warmup_cosine_scheduler(optimizer)
    
    use_cuda = torch.cuda.is_available() and config.DEVICE.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # 7. Training Loop with Progressive Unfreezing and Early Stopping
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'train_auc': []}
    best_val_auc = 0.0
    patience_counter = 0
    unfrozen_epochs = set()

    print("\n=== STARTING TRAINING PIPELINE ===")
    for epoch in range(1, config.EPOCHS + 1):
        # Apply progressive unfreezing at epoch marks
        handle_progressive_unfreeze(epoch, model, optimizer, unfrozen_epochs)

        # Run training epoch
        train_loss, train_auc, _, _ = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, scaler=scaler, device=config.DEVICE, training=True
        )

        # Run validation epoch
        val_loss, val_auc, _, _ = run_epoch(
            model=model, loader=val_loader, optimizer=None,
            criterion=criterion, scaler=scaler, device=config.DEVICE, training=False
        )
        
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['train_auc'].append(train_auc)

        # Save check
        saved_flag = ''
        if val_auc > (best_val_auc + config.EARLY_STOP_MIN_DELTA):
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            saved_flag = '  ✓ saved'
        else:
            patience_counter += 1

        bb_lr = optimizer.param_groups[0]['lr']
        hd_lr = optimizer.param_groups[3]['lr']
        print(f"Epoch [{epoch:3d}/{config.EPOCHS}]  "
              f"TrLoss: {train_loss:.4f}  VaLoss: {val_loss:.4f}  "
              f"TrAUC: {train_auc:.4f}  VaAUC: {val_auc:.4f}  "
              f"LR(bb/hd): {bb_lr:.1e}/{hd_lr:.1e}  "
              f"Patience: {patience_counter}/{config.EARLY_STOP_PATIENCE}"
              f"{saved_flag}")

        # Check early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}. Best Val AUC: {best_val_auc:.4f}")
            break

    print(f"\nTraining completed. Best Validation AUC: {best_val_auc:.4f}")

    # 8. Plot training curves
    plot_training_curves(history, 'training_curves_green_multimodal_v2.png')

if __name__ == '__main__':
    main()
