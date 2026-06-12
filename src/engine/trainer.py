import numpy as np
import torch
import torch.cuda.amp as amp
from sklearn.metrics import roc_auc_score
from configs import config

def mixup_data(right, left, clinical, labels, alpha=0.2, device='cpu'):
    """Applies Mixup data augmentation to inputs and returns mixed variables and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
        
    B = right.size(0)
    idx = torch.randperm(B, device=device)
    
    mixed_right = lam * right + (1 - lam) * right[idx]
    mixed_left = lam * left + (1 - lam) * left[idx]
    mixed_clinical = lam * clinical + (1 - lam) * clinical[idx]
    labels_a, labels_b = labels, labels[idx]
    
    return mixed_right, mixed_left, mixed_clinical, labels_a, labels_b, lam


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    """Computes weighted loss for Mixup data samples."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def handle_progressive_unfreeze(epoch, model, optimizer, unfrozen_epochs):
    """
    Progressively unfreezes layers in ViT backbone at configured epoch marks.
    Adds a new parameter group with custom learning rate.
    """
    if epoch in config.UNFREEZE_SCHEDULE and epoch not in unfrozen_epochs:
        start_blk, end_blk = config.UNFREEZE_SCHEDULE[epoch]
        blk_slice = model.backbone.blocks[start_blk:end_blk]
        
        # Enable gradients for the newly unfrozen blocks
        for p in blk_slice.parameters():
            p.requires_grad = True
            
        optimizer.add_param_group({
            'params': [p for p in blk_slice.parameters() if p.requires_grad],
            'lr': config.BACKBONE_LR * 0.5,
        })
        
        unfrozen_epochs.add(epoch)
        print(f'  ▶ Epoch {epoch}: unfrozen backbone blocks[{start_blk}:{end_blk}]  (LR={config.BACKBONE_LR*0.5:.1e})')


def run_epoch(model, loader, optimizer, criterion, scaler, device, training=True):
    """
    Runs one training or validation epoch over the dataloader.
    """
    if training:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    all_probs = []
    all_lbls = []

    context = torch.enable_grad() if training else torch.no_grad()
    use_cuda = torch.cuda.is_available() and device.type == 'cuda'
    
    with context:
        for right, left, clinical, labels in loader:
            right = right.to(device, non_blocking=True)
            left = left.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

                if config.USE_MIXUP and np.random.rand() < 0.5:
                    mixed_r, mixed_l, mixed_c, la, lb, lam = mixup_data(
                        right, left, clinical, labels, alpha=config.MIXUP_ALPHA, device=device
                    )
                    with amp.autocast(enabled=use_cuda):
                        logits = model(mixed_r, mixed_l, mixed_c)
                        loss = mixup_criterion(criterion, logits, la, lb, lam)
                    
                    # Track metrics using clean non-augmented validation pass outputs
                    with torch.no_grad():
                        probs_track = torch.sigmoid(model(right, left, clinical))
                else:
                    with amp.autocast(enabled=use_cuda):
                        logits = model(right, left, clinical)
                        loss = criterion(logits, labels)
                    probs_track = torch.sigmoid(logits).detach()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

                all_probs.extend(probs_track.cpu().numpy())
                all_lbls.extend(labels.cpu().numpy())
            else:
                with amp.autocast(enabled=use_cuda):
                    logits = model(right, left, clinical)
                    loss = criterion(logits, labels)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.extend(probs)
                all_lbls.extend(labels.cpu().numpy())

            loss_sum += loss.item()

    avg_loss = loss_sum / len(loader)
    auc = roc_auc_score(all_lbls, all_probs) if len(set(all_lbls)) > 1 else 0.0
    return avg_loss, auc, np.array(all_probs), np.array(all_lbls)
