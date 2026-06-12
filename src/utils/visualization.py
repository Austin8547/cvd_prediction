import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, brier_score_loss, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image
from configs import config
from .xai import RightEyeWrapper, LeftEyeWrapper, compute_gradcam

# Normalization parameters for restoring RGB image visualization
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def tensor_to_rgb(tensor):
    """Restores a normalized PyTorch tensor image back to a denormalized float32 RGB range."""
    img = tensor[0].cpu().numpy().transpose(1, 2, 0)
    return (img * STD + MEAN).clip(0, 1)


def load_eye_image(filename, image_folder, device):
    """Helper to read, resize, and normalize an image to match base inputs."""
    path = os.path.join(image_folder, filename)
    img = read_image(path)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3]
        
    base_tf = T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return base_tf(img).unsqueeze(0).to(device)


def plot_training_curves(history, output_path):
    """Generates and saves the training history curves (Loss, Val/Train AUC, and overfit monitors)."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss curve
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', markersize=2)
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss', marker='s', markersize=2)
    axes[0].set_title('Training & Validation Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # AUC Curve
    axes[1].plot(epochs_range, history['train_auc'], label='Train AUC', color='orange', marker='o', markersize=2)
    axes[1].plot(epochs_range, history['val_auc'], label='Val AUC', color='green', marker='^', markersize=2)
    axes[1].axhline(y=0.8279, color='gray', linestyle='--', alpha=0.7, label='Paper Unimodal 82.79%')
    axes[1].axhline(y=0.8601, color='salmon', linestyle='--', alpha=0.7, label='Paper Multimodal 86.01%')
    best_val_auc = max(history['val_auc'])
    axes[1].axhline(y=best_val_auc, color='blue', linestyle=':', alpha=0.8, label=f'Best Val = {best_val_auc:.4f}')
    axes[1].set_title('AUC-ROC Over Training', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Gap monitor
    if len(history['train_auc']) == len(history['val_auc']):
        gap = [t - v for t, v in zip(history['train_auc'], history['val_auc'])]
        axes[2].plot(epochs_range, gap, color='red', marker='.', markersize=2)
        axes[2].axhline(y=0, color='black', linestyle='--', lw=1)
        axes[2].axhline(y=0.05, color='orange', linestyle=':', lw=1, label='5% gap warn')
        axes[2].set_title('Train-Val AUC Gap (Overfit Monitor)', fontsize=13)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Gap')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Green Multimodal Siamese (RETFound-Green) - Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to: {output_path}')


def plot_calibration_and_pr(labels, probs, optimal_thresh, output_path):
    """Generates and saves model calibration, probability distribution, and PR curves."""
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10, strategy='uniform')
    brier = brier_score_loss(labels, probs)

    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    closest_idx = np.argmin(np.abs(pr_thresholds - optimal_thresh))
    opt_precision = precision[closest_idx]
    opt_recall = recall[closest_idx]
    baseline = labels.mean()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Calibration Curve
    axes[0].plot([0, 1], [0, 1], linestyle='--', lw=2, label='Perfect')
    axes[0].plot(prob_pred, prob_true, marker='o', lw=2, label='Model')
    axes[0].set_title(f'Calibration Curve\n(Brier = {brier:.4f})')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].legend()
    axes[0].grid(True)

    # Prob distribution
    axes[1].hist(probs, bins=50)
    axes[1].axvline(optimal_thresh, linestyle='--', lw=2, label=f'Threshold = {optimal_thresh:.3f}')
    axes[1].set_title('Probability Distribution')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True)

    # PR Curve
    axes[2].plot(recall, precision, lw=2, label=f'PR Curve (AP = {ap:.4f})')
    axes[2].axhline(y=baseline, linestyle='--', lw=1.5, label=f'Baseline = {baseline:.2f}')
    axes[2].scatter(opt_recall, opt_precision, marker='*', s=200, zorder=5,
                    label=f'Threshold = {optimal_thresh:.3f}\nP = {opt_precision:.3f}, R = {opt_recall:.3f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('Validation Set - Calibration & PR Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Calibration/PR analysis saved to: {output_path}')


def plot_confusion_and_roc(labels, probs, preds, optimal_thresh, output_path):
    """Generates and saves the ROC Curve and Confusion Matrix plots."""
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f'Confusion Matrix (Threshold = {optimal_thresh:.4f})', fontsize=13)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, lw=2, color='steelblue', label=f'Model (AUC = {auc_val:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1].axhline(y=0.9041, color='green', linestyle=':', alpha=0.7, label='Paper Multimodal AUC=90.41%')
    axes[1].axhline(y=0.8258, color='orange', linestyle=':', alpha=0.7, label='Paper Unimodal AUC=82.58%')
    axes[1].scatter([1 - specificity], [sensitivity], color='blue', zorder=5, s=80, label='Operating point')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve (Test Set)', fontsize=13)
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.suptitle('Model Test Set Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'ROC/Confusion plots saved to: {output_path}')


def generate_single_gradcam(model, sample_row, image_folder, target_layer, optimal_thresh, output_path, device):
    """Generates and saves the GradCAM++ visualization for a single patient sample."""
    model.eval()
    
    right_t = load_eye_image(sample_row['right_eye'], image_folder, device)
    left_t = load_eye_image(sample_row['left_eye'], image_folder, device)
    
    clinical_t = torch.tensor([[
        float(sample_row['age_norm']),
        float(sample_row['gender_female']),
        float(sample_row['gender_male']),
    ]], dtype=torch.float32).to(device)

    # Freeze context to run XAI on target eyes
    with torch.no_grad():
        fixed_left_feat = model.encode_image(left_t)
        fixed_right_feat = model.encode_image(right_t)
        fixed_clin_feat = model.encode_clinical(clinical_t)
        pred_prob = torch.sigmoid(model(right_t, left_t, clinical_t)).item()
        pred_label = int(pred_prob >= optimal_thresh)
        
    gt_label = int(sample_row['label'])

    # Wrap models for separate eye GradCAM runs
    right_wrapper = RightEyeWrapper(model, fixed_left_feat.detach(), fixed_clin_feat.detach()).to(device)
    left_wrapper = LeftEyeWrapper(model, fixed_right_feat.detach(), fixed_clin_feat.detach()).to(device)

    right_overlay, _ = compute_gradcam(right_wrapper, right_t, target_layer)
    left_overlay, _ = compute_gradcam(left_wrapper, left_t, target_layer)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f'GradCAM++ - Patient {sample_row["patient_id"]} '
        f'(Age={sample_row.get("true_age", "N/A")}, '
        f'{"Male" if sample_row["gender_male"]==1 else "Female"})\n'
        f'GT: {config.CLASS_NAMES[gt_label]}  |  Pred: {config.CLASS_NAMES[pred_label]} (prob={pred_prob:.4f})',
        fontsize=13, fontweight='bold'
    )
    
    axes[0, 0].imshow(tensor_to_rgb(right_t))
    axes[0, 0].set_title('Right Eye - Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_overlay)
    axes[0, 1].set_title('Right Eye - GradCAM++')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(tensor_to_rgb(left_t))
    axes[1, 0].set_title('Left Eye - Original')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(left_overlay)
    axes[1, 1].set_title('Left Eye - GradCAM++')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Single GradCAM overlay grid saved to: {output_path}')


def generate_multi_gradcam(model, test_df, image_folder, target_layer, optimal_thresh, output_path, device, n_samples=3):
    """Generates and saves the GradCAM++ visualizations for multiple patient samples in a grid."""
    model.eval()
    
    fig2, axes2 = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    fig2.suptitle('GradCAM++ overlays - Multiple Test Samples', fontsize=14, fontweight='bold')

    col_titles = ['Right Eye (Original)', 'Right Eye (GradCAM++)',
                  'Left Eye (Original)', 'Left Eye (GradCAM++)']
                  
    for ax, title in zip(axes2[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight='bold')

    for i in range(n_samples):
        row = test_df.iloc[i]
        gt_l = int(row['label'])
        
        r_t = load_eye_image(row['right_eye'], image_folder, device)
        l_t = load_eye_image(row['left_eye'], image_folder, device)
        c_t = torch.tensor([[
            float(row['age_norm']),
            float(row['gender_female']),
            float(row['gender_male'])
        ]], dtype=torch.float32).to(device)

        with torch.no_grad():
            fl = model.encode_image(l_t)
            fr = model.encode_image(r_t)
            fc = model.encode_clinical(c_t)
            p = torch.sigmoid(model(r_t, l_t, c_t)).item()
            pl = int(p >= optimal_thresh)

        # Create GradCAM wrappers
        rw = RightEyeWrapper(model, fl.detach(), fc.detach()).to(device)
        lw = LeftEyeWrapper(model, fr.detach(), fc.detach()).to(device)
        
        r_ov, _ = compute_gradcam(rw, r_t, target_layer)
        l_ov, _ = compute_gradcam(lw, l_t, target_layer)

        axes2[i, 0].imshow(tensor_to_rgb(r_t))
        axes2[i, 1].imshow(r_ov)
        axes2[i, 2].imshow(tensor_to_rgb(l_t))
        axes2[i, 3].imshow(l_ov)

        correct = '✓' if gt_l == pl else '✗'
        age_disp = row.get('true_age', 'N/A')
        sex_disp = 'M' if row['gender_male'] == 1 else 'F'
        
        row_label = f'{correct} GT:{config.CLASS_NAMES[gt_l]}\nPred:{config.CLASS_NAMES[pl]}\n(p={p:.3f})\nAge={age_disp} {sex_disp}'
        axes2[i, 0].set_ylabel(row_label, fontsize=8, rotation=0, labelpad=80, va='center')
        
        for ax in axes2[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Multi-sample GradCAM overlay grid saved to: {output_path}')
