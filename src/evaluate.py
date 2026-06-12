import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
    classification_report
)
import sys

# Add current file's directory (src/) to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from configs import config
from data import get_dataloaders
from models import GreenMultimodalSiamese
from engine import run_inference, search_optimal_thresholds, run_clinical_ablation
from utils import plot_calibration_and_pr, plot_confusion_and_roc


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
        })
    df = pd.DataFrame(records)

    val_df = df[df['group'] == 2].reset_index(drop=True)
    test_df = df[df['group'] == 3].reset_index(drop=True)

    print(f"Val size: {len(val_df)}, Test size: {len(test_df)}")

    # 2. Datasets & Loaders
    _, val_loader, test_loader, test_tta_loader = get_dataloaders(
        train_df=None, val_df=val_df, test_df=test_df,
        image_folder=config.IMAGE_FOLDER
    )

    # 3. Load Model
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

    # 4. Deterministic and TTA Inference
    print("\n--- Running Validation Inference (Deterministic) ---")
    val_probs_det, val_labels = run_inference(model, val_loader, config.DEVICE, passes=1)

    print("\n--- Running Test Inference (Deterministic) ---")
    test_probs_det, test_labels = run_inference(model, test_loader, config.DEVICE, passes=1)

    print(f"\n--- Running Test Inference (TTA, {config.TTA_N_PASSES} passes) ---")
    test_probs_tta, _ = run_inference(model, test_tta_loader, config.DEVICE, passes=config.TTA_N_PASSES)

    # 5. Grid Search Temperature Calibration & Optimization
    temp_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    print("\n" + "="*80)
    print("      DETERMINISTIC VALIDATION CALIBRATION GRID SEARCH")
    print("="*80)
    det_opt_df, best_det_row = search_optimal_thresholds(val_probs_det, val_labels, temp_grid)
    print(det_opt_df.to_string(index=False))
    print(f"\nBest Det Temp: T = {best_det_row['T']:.2f} (Brier: {best_det_row['Brier']:.4f})")

    # Generate calibration outputs on validation set
    plot_calibration_and_pr(
        labels=val_labels,
        probs=val_probs_det,
        optimal_thresh=best_det_row['Thresh_Youden'],
        output_path='validation_analysis_green_v2.png'
    )

    # 6. Apply Calibration and Thresholds to Test Set
    eps = 1e-7
    t_det = best_det_row['T']
    
    # Scale test probabilities using the validation-optimized temperatures
    det_test_scaled = expit(logit(np.clip(test_probs_det, eps, 1.0-eps)) / t_det)
    
    # We use same temperature and threshold configurations for standard & TTA evaluations
    eval_configs = [
        # Deterministic Evaluations (thresholds selected on validation)
        ("Deterministic (T=1.0)", test_probs_det, best_det_row['Thresh_Youden'], "Youden J", "Baseline (Deterministic, Balanced)"),
        ("Deterministic (T=1.0)", test_probs_det, best_det_row['Thresh_Sens85'], "Sens >= 85%", "Raw High-Sens Deterministic"),
        (f"Deterministic (T={t_det:.1f})", det_test_scaled, best_det_row['Thresh_Youden'], "Youden J", "Calibrated Deterministic Balanced"),
        (f"Deterministic (T={t_det:.1f})", det_test_scaled, best_det_row['Thresh_Sens85'], "Sens >= 85%", "Calibrated Deterministic High-Sens"),
        (f"Deterministic (T={t_det:.1f})", det_test_scaled, best_det_row['Thresh_F2'], "F2 Score", "Calibrated Deterministic F2"),
        
        # TTA 5-Passes Evaluations
        (f"TTA {config.TTA_N_PASSES}-Passes (T=1.0)", test_probs_tta, best_det_row['Thresh_Youden'], "Youden J", "TTA Balanced"),
        (f"TTA {config.TTA_N_PASSES}-Passes (T=1.0)", test_probs_tta, best_det_row['Thresh_Sens85'], "Sens >= 85%", "TTA High-Sens (Clinical Target)"),
    ]

    test_evals = []
    for mode, probs, thresh, thresh_type, label in eval_configs:
        preds = (probs >= thresh).astype(int)
        
        acc = accuracy_score(test_labels, preds)
        precision = precision_score(test_labels, preds, zero_division=0)
        recall = recall_score(test_labels, preds, zero_division=0)
        f1 = f1_score(test_labels, preds, zero_division=0)
        auc = roc_auc_score(test_labels, probs)
        brier = brier_score_loss(test_labels, probs)
        
        macro_r = recall_score(test_labels, preds, average='macro', zero_division=0)
        macro_f1 = f1_score(test_labels, preds, average='macro', zero_division=0)
        spec = recall_score(test_labels, preds, pos_label=0, zero_division=0)
        
        test_evals.append({
            'Config Label': label,
            'Inference Mode': mode,
            'Selected Threshold': f"{thresh:.4f} ({thresh_type})",
            'Accuracy': acc,
            'Sensitivity (Recall)': recall,
            'Specificity': spec,
            'F1 Score': f1,
            'Macro Recall': macro_r,
            'Macro F1': macro_f1,
            'AUC-ROC': auc,
            'Brier Score': brier
        })

    test_eval_df = pd.DataFrame(test_evals)
    
    print("\n" + "="*120)
    print("                                      TEST SET EVALUATION RESULTS")
    print("="*120)
    print(test_eval_df.to_string(index=False))
    print("="*120)

    # 7. Print classification reports for key configurations
    print("\n" + "#"*70)
    print("  DETAILED CLASSIFICATION REPORT FOR KEY CONFIGURATIONS")
    print("#"*70)

    # We will plotconfusion matrix and ROC for the standard Calibrated Deterministic Balanced model
    opt_thresh_final = best_det_row['Thresh_Youden']
    opt_preds_final = (det_test_scaled >= opt_thresh_final).astype(int)

    plot_confusion_and_roc(
        labels=test_labels,
        probs=det_test_scaled,
        preds=opt_preds_final,
        optimal_thresh=opt_thresh_final,
        output_path='evaluation_green_multimodal_v2.png'
    )

    for mode, probs, thresh, thresh_type, label in [
        ("Deterministic (T=1.0)", test_probs_det, best_det_row['Thresh_Youden'], "Youden J", "Baseline (No Calibration, Balanced)"),
        (f"Deterministic (T={t_det:.1f})", det_test_scaled, best_det_row['Thresh_Sens85'], "Sens >= 85%", "Calibrated Deterministic High-Sensitivity"),
        (f"TTA {config.TTA_N_PASSES}-Passes (T=1.0)", test_probs_tta, best_det_row['Thresh_Sens85'], "Sens >= 85%", "TTA High-Sensitivity")
    ]:
        preds = (probs >= thresh).astype(int)
        print(f"\nConfiguration: {label}")
        print(f"  Threshold : {thresh:.4f} ({thresh_type})")
        print(f"  AUC-ROC   : {roc_auc_score(test_labels, probs):.4f}")
        print(classification_report(test_labels, preds, target_names=config.CLASS_NAMES))

    # 8. Clinical Feature Ablation Analysis
    print("\n--- Running Clinical Feature Ablation study on Test Set ---")
    ablation_results = run_clinical_ablation(model, test_loader, config.DEVICE, opt_thresh_final)

    print('=' * 55)
    print(f'  {"Scenario":<25}  {"AUC":>7}  {"Acc":>7}  {"F1":>7}')
    print('-' * 55)
    for name, metrics in ablation_results.items():
        print(f'  {name:<25}  {metrics["AUC"]:>7.4f}  {metrics["Accuracy"]:>7.4f}  {metrics["F1"]:>7.4f}')
    print('=' * 55)

    # Plot ablation horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(ablation_results.keys())
    aucs = [ablation_results[n]['AUC'] for n in names]
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    bars = ax.barh(names, aucs, color=colors, edgecolor='white')
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Clinical Feature Ablation - green_multi_v2', fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('clinical_ablation_green.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Clinical ablation study plot saved: clinical_ablation_green.png')

if __name__ == '__main__':
    main()
