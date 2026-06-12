import os
import json
import pandas as pd
import torch
import sys

# Add current file's directory (src/) to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from configs import config
from models import GreenMultimodalSiamese
from utils import generate_single_gradcam, generate_multi_gradcam


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
        
        records.append({
            'patient_id': patient_id,
            'right_eye': info['right_eye'],
            'left_eye': info['left_eye'],
            'label': int(info['label']),
            'group': int(info['group']),
            'gender_raw': gender_raw,
            'gender_female': gender_onehot[0],
            'gender_male': gender_onehot[1],
            'age_norm': float(info.get('age', 0.0)),
            'true_age': info.get('True_age', None),
        })
    df = pd.DataFrame(records)

    test_df = df[df['group'] == 3].reset_index(drop=True)
    print(f"Loaded test dataframe with {len(test_df)} samples.")

    # 2. Load Model
    model = GreenMultimodalSiamese(
        weights_path=config.WEIGHTS_PATH,
        img_size=config.IMG_SIZE,
        clinical_in_dim=3,
        clinical_feat_dim=config.CLINICAL_FEAT_DIM
    ).to(config.DEVICE)
    
    print(f"Loading checkpoint from {config.BEST_MODEL_PATH}...")
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {config.BEST_MODEL_PATH}. Run train.py first.")
    
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))

    # 3. Hook the target ViT layer for GradCAM calculations
    # Standard target layer for vit_base_patch16_224 in PyTorch GradCAM is the normalization layer inside the final block.
    target_layer = model.backbone.blocks[-1].norm1
    print(f"Hooked Target ViT layer: {target_layer}")

    # 4. Generate Single Patient GradCAM Overlay Grid
    print("\nGenerating Single Patient GradCAM overlay grid...")
    sample_patient = test_df.iloc[0]
    
    # We use a default threshold of 0.5 or 0.380 (Youden's typical threshold) for display predictions
    optimal_thresh = 0.5 
    
    generate_single_gradcam(
        model=model,
        sample_row=sample_patient,
        image_folder=config.IMAGE_FOLDER,
        target_layer=target_layer,
        optimal_thresh=optimal_thresh,
        output_path='gradcam_single_patient.png',
        device=config.DEVICE
    )

    # 5. Generate Multi Patient GradCAM Overlay Grid
    print("\nGenerating multi-patient GradCAM overlay grids...")
    generate_multi_gradcam(
        model=model,
        test_df=test_df,
        image_folder=config.IMAGE_FOLDER,
        target_layer=target_layer,
        optimal_thresh=optimal_thresh,
        output_path='gradcam_multi_patients.png',
        device=config.DEVICE,
        n_samples=3
    )

    print("\nGradCAM++ visualizations generated successfully!")
    print("Files saved:")
    print("  - Single Patient: gradcam_single_patient.png")
    print("  - Multi Patient Grid: gradcam_multi_patients.png")

if __name__ == '__main__':
    main()
