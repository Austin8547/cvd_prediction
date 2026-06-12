import os
import math
import argparse
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.v2 as T
from torchvision.io import read_image
from configs import config

# Offline metadata path inside our augmented data directory
AUG_METADATA_PATH = os.path.join(config.AUG_SAVE_DIR, 'augmented_train_metadata_green_multimodal.csv')

def augment_and_save(src_filename, save_prefix, aug_id, aug_pipeline, image_folder, save_dir):
    """
    Reads an image, applies the augmentation pipeline, and saves the output to disk.
    """
    src_path = os.path.join(image_folder, src_filename)
    
    # Read as uint8 tensor
    img_uint8 = read_image(src_path)
    
    # Normalize channels
    if img_uint8.shape[0] == 1:
        img_uint8 = img_uint8.repeat(3, 1, 1)
    elif img_uint8.shape[0] == 4:
        img_uint8 = img_uint8[:3]
        
    # Apply augmentations (maintaining uint8 tensor structure for easy conversion to PIL)
    aug_tensor = aug_pipeline(img_uint8)
    
    # Permute to HWC and convert to numpy for PIL
    pil_img = Image.fromarray(aug_tensor.permute(1, 2, 0).numpy())
    
    save_name = f'{save_prefix}_aug{aug_id}.png'
    pil_img.save(os.path.join(save_dir, save_name))
    return save_name


def run_offline_augmentation(train_df, image_folder, save_dir, img_size=config.IMG_SIZE, force_rerun=False):
    """
    Balances the dataset by augmenting the minority/both classes to reach the target size.
    Returns:
        pd.DataFrame: Augmented train dataframe containing paths to new files and matching clinical data.
    """
    if os.path.exists(AUG_METADATA_PATH) and not force_rerun:
        print(f"Loading existing augmentation metadata from {AUG_METADATA_PATH}...")
        augmented_train_df = pd.read_csv(AUG_METADATA_PATH)
        augmented_train_df['is_augmented'] = augmented_train_df['is_augmented'].astype(bool)
        return augmented_train_df

    print("Running offline data augmentation...")
    os.makedirs(save_dir, exist_ok=True)

    n_normal_tr = (train_df['label'] == 0).sum()
    n_thickened_tr = (train_df['label'] == 1).sum()
    
    # Target size: double the size of the majority class
    target_per_class = max(n_normal_tr, n_thickened_tr) * 2

    print(f"Original Train Balance: Normal={n_normal_tr}, Thickened={n_thickened_tr}")
    print(f"Target size: {target_per_class} samples per class (~{target_per_class * 2:,} total images).")

    # Medically-safe augmentation pipeline
    offline_aug_pipeline = T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3),
    ])

    aug_rows = []

    for label in [0, 1]:
        class_df = train_df[train_df['label'] == label].reset_index(drop=True)
        current_count = len(class_df)

        if current_count >= target_per_class:
            print(f"Class {label}: already at {current_count} >= {target_per_class}, skipping.")
            continue

        needed = target_per_class - current_count
        copies_per_sample = math.ceil(needed / current_count)

        print(f"Augmenting Class {label} ({current_count} -> {target_per_class}) using up to {copies_per_sample} copies per patient...")
        
        generated_for_class = 0

        for _, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"Class {label} augmentation"):
            pid = str(row['patient_id']).replace('/', '_')

            for aug_id in range(1, copies_per_sample + 1):
                if (current_count + generated_for_class) >= target_per_class:
                    break

                aug_right = augment_and_save(
                    row['right_eye'], f'{pid}_R', aug_id,
                    offline_aug_pipeline, image_folder, save_dir
                )
                aug_left = augment_and_save(
                    row['left_eye'], f'{pid}_L', aug_id,
                    offline_aug_pipeline, image_folder, save_dir
                )

                aug_rows.append({
                    'patient_id': f'{row["patient_id"]}_aug{aug_id}',
                    'right_eye': aug_right,
                    'left_eye': aug_left,
                    'label': label,
                    'group': 1,
                    'is_augmented': True,
                    'gender_raw': row['gender_raw'],
                    'gender_female': row['gender_female'],
                    'gender_male': row['gender_male'],
                    'age_norm': row['age_norm'],
                    'true_age': row.get('true_age', None),
                    'thickness': row.get('thickness', None),
                })
                generated_for_class += 1

    # Combine original training set with the augmented copies
    train_df_copy = train_df.copy()
    train_df_copy['is_augmented'] = False
    
    augmented_train_df = pd.concat([train_df_copy, pd.DataFrame(aug_rows)], ignore_index=True)
    
    # Shuffle the combined dataset
    augmented_train_df = augmented_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save metadata
    augmented_train_df.to_csv(AUG_METADATA_PATH, index=False)
    print(f"Augmentation complete: {len(aug_rows)} new records generated and saved to {AUG_METADATA_PATH}.")

    na = (augmented_train_df['label'] == 0).sum()
    th = (augmented_train_df['label'] == 1).sum()
    print(f"Final training set balance: Normal={na} | Thickened={th} | Total={len(augmented_train_df)}")

    return augmented_train_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline training image augmentation.")
    parser.add_argument('--force', action='store_true', help="Force regeneration of augmented dataset.")
    args = parser.parse_args()

    # Load original data splits to get train_df
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
            'thickness': info.get('thickness', None),
        })

    df = pd.DataFrame(records)
    train_df = df[df['group'] == 1].reset_index(drop=True)

    run_offline_augmentation(
        train_df=train_df,
        image_folder=config.IMAGE_FOLDER,
        save_dir=config.AUG_SAVE_DIR,
        img_size=config.IMG_SIZE,
        force_rerun=args.force
    )
