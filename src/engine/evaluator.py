import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import logit, expit
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, brier_score_loss
)
from configs import config

def run_inference(model, loader, device, passes=1):
    """
    Runs evaluation on a model, supporting both standard single pass
    and Test-Time Augmentation (TTA) multi-pass averaging.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for p in range(passes):
            pass_probs = []
            pass_labels = []
            
            # Use tqdm only if more than 1 pass or larger loader
            desc = f"Inference Pass {p+1}/{passes}" if passes > 1 else "Inference"
            for right, left, clinical, labels in tqdm(loader, desc=desc, leave=False):
                right = right.to(device)
                left = left.to(device)
                clinical = clinical.to(device)
                
                outputs = torch.sigmoid(model(right, left, clinical)).cpu().numpy()
                pass_probs.extend(outputs)
                if p == 0:
                    pass_labels.extend(labels.numpy())
            
            all_probs.append(pass_probs)
            if p == 0:
                all_labels = pass_labels

    # Average probabilities across TTA passes
    mean_probs = np.mean(all_probs, axis=0)
    return mean_probs, np.array(all_labels)


def search_optimal_thresholds(probs, labels, temp_grid):
    """
    Performs grid search on temperature calibration values and computes optimal decision
    thresholds under multiple clinical validation strategies:
      1. Youden's J Statistic (Balanced accuracy)
      2. Macro F1 Score
      3. Sensitivity-Prioritized (Guarantees >= 85% recall, maximizes specificity)
      4. F2 Score (Prioritizes recall twice as much as precision)
    """
    eps = 1e-7
    # Convert probabilities to logits for temperature scaling
    logits = logit(np.clip(probs, eps, 1.0 - eps))

    records = []
    for T in temp_grid:
        scaled_probs = expit(logits / T)
        brier = brier_score_loss(labels, scaled_probs)
        auc = roc_auc_score(labels, scaled_probs)

        # Strategy 1: Youden's J
        fpr, tpr, thresholds = roc_curve(labels, scaled_probs)
        youden = tpr - fpr
        best_j_idx = np.argmax(youden)
        thresh_j = thresholds[best_j_idx]

        # Strategy 2: Macro F1
        best_f1 = -1
        thresh_f1 = 0.5
        for th in np.linspace(0.01, 0.99, 100):
            th_preds = (scaled_probs >= th).astype(int)
            score = f1_score(labels, th_preds, average='macro', zero_division=0)
            if score > best_f1:
                best_f1 = score
                thresh_f1 = th

        # Strategy 3: Sensitivity-Prioritized (guarantees validation recall >= 85%)
        thresh_sens = 0.5
        best_spec_at_sens = -1
        found_sens = False
        for th in np.linspace(0.01, 0.99, 100):
            th_preds = (scaled_probs >= th).astype(int)
            sens = recall_score(labels, th_preds, zero_division=0)
            if sens >= 0.85:
                spec = recall_score(labels, th_preds, pos_label=0, zero_division=0)
                if spec > best_spec_at_sens:
                    best_spec_at_sens = spec
                    thresh_sens = th
                    found_sens = True
        if not found_sens:
            thresh_sens = thresh_j

        # Strategy 4: F2 Score
        best_f2 = -1
        thresh_f2 = 0.5
        for th in np.linspace(0.01, 0.99, 100):
            th_preds = (scaled_probs >= th).astype(int)
            p = precision_score(labels, th_preds, zero_division=0)
            r = recall_score(labels, th_preds, zero_division=0)
            if (4 * p + r) > 0:
                f2 = 5 * p * r / (4 * p + r)
            else:
                f2 = 0
            if f2 > best_f2:
                best_f2 = f2
                thresh_f2 = th

        records.append({
            'T': T,
            'Brier': brier,
            'AUC': auc,
            'Thresh_Youden': thresh_j,
            'Thresh_MacroF1': thresh_f1,
            'Thresh_Sens85': thresh_sens,
            'Thresh_F2': thresh_f2,
        })

    df = pd.DataFrame(records)
    best_row = df.loc[df['Brier'].idxmin()]
    return df, best_row


def run_clinical_ablation(model, loader, device, optimal_thresh):
    """
    Performs ablation study by masking age and gender inputs.
    Scenarios:
      - Full (Age + Gender) -> [1, 1, 1]
      - Age only            -> [1, 0, 0]
      - Gender only         -> [0, 1, 1]
      - No clinical (zeros) -> [0, 0, 0]
    """
    scenarios = {
        'Full (age + gender)': [1.0, 1.0, 1.0],
        'Age only':            [1.0, 0.0, 0.0],
        'Gender only':         [0.0, 1.0, 1.0],
        'No clinical (zeros)': [0.0, 0.0, 0.0],
    }

    model.eval()
    results = {}

    for name, mask in scenarios.items():
        probs_out = []
        lbls_out = []
        mask_t = torch.tensor(mask, dtype=torch.float32).to(device)

        with torch.no_grad():
            for right, left, clinical, labels in loader:
                right = right.to(device)
                left = left.to(device)
                clinical = clinical.to(device)
                
                # Apply mask to features
                clinical_masked = clinical * mask_t
                
                logits = model(right, left, clinical_masked)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_out.extend(probs)
                lbls_out.extend(labels.numpy())

        p = np.array(probs_out)
        l = np.array(lbls_out)
        preds = (p >= optimal_thresh).astype(int)

        results[name] = {
            'AUC': roc_auc_score(l, p),
            'Accuracy': accuracy_score(l, preds),
            'F1': f1_score(l, preds, zero_division=0),
        }

    return results
