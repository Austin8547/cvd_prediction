import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision.io import read_image
from configs import config

class MultimodalFundusDataset(Dataset):
    """
    Custom PyTorch Dataset for Multimodal Siamese Fundus Image + Clinical Features.
    Returns:
        right_eye_tensor (Tensor): Preprocessed right eye image.
        left_eye_tensor (Tensor): Preprocessed left eye image.
        clinical_tensor (Tensor): Normalized age and gender features [age, gender_female, gender_male].
        label_tensor (Tensor): Binary label for CIMT (Normal vs Thickened).
    """
    def __init__(self, dataframe, image_folder, aug_folder=None, is_train=False, tta_mode=False):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.aug_folder = aug_folder
        self.is_train = is_train
        self.tta_mode = tta_mode

        # Medically-safe training augmentation pipeline
        self.train_transforms = T.Compose([
            T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Test-Time Augmentation (TTA) pipeline
        self.tta_transforms = T.Compose([
            T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.05, contrast=0.05),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Standard baseline validation/test transform
        self.base_transforms = T.Compose([
            T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_image(self, filename, is_augmented=False):
        """Helper to read, normalize channel counts, and transform fundus images."""
        if is_augmented and self.aug_folder:
            path = os.path.join(self.aug_folder, filename)
        else:
            path = os.path.join(self.image_folder, filename)

        img = read_image(path)
        
        # Ensure 3 channels (grayscale replication, transparency drop)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]

        # Apply appropriate transformation pipeline
        if self.is_train:
            return self.train_transforms(img)
        elif self.tta_mode:
            return self.tta_transforms(img)
        return self.base_transforms(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        is_aug = bool(row.get('is_augmented', False))
        
        right = self._load_image(row['right_eye'], is_augmented=is_aug)
        left = self._load_image(row['left_eye'], is_augmented=is_aug)

        clinical = torch.tensor([
            float(row['age_norm']),
            float(row['gender_female']),
            float(row['gender_male']),
        ], dtype=torch.float32)

        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return right, left, clinical, label


def get_dataloaders(train_df, val_df, test_df, image_folder, aug_folder=None, batch_size=None, num_workers=None):
    """
    Helper to construct training, validation, standard test, and TTA test dataloaders.
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    nw = num_workers if num_workers is not None else config.NUM_WORKERS

    train_dataset = MultimodalFundusDataset(train_df, image_folder, aug_folder=aug_folder, is_train=True)
    val_dataset = MultimodalFundusDataset(val_df, image_folder, is_train=False)
    test_dataset = MultimodalFundusDataset(test_df, image_folder, is_train=False)
    test_tta_dataset = MultimodalFundusDataset(test_df, image_folder, is_train=False, tta_mode=True)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, 
        num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, 
        num_workers=nw, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, 
        num_workers=nw, pin_memory=True
    )
    test_tta_loader = DataLoader(
        test_tta_dataset, batch_size=bs, shuffle=False, 
        num_workers=nw, pin_memory=True
    )

    return train_loader, val_loader, test_loader, test_tta_loader
