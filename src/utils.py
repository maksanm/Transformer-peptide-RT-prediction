import random

import torch
from torch.utils.data import Dataset
from dataset import collate
from scipy.stats import spearmanr
import numpy as np


def split_dataset(ds: Dataset, val_ratio=0.05, seed=42):
    """
    Split dataset into train/val subsets.
    """
    N = len(ds)
    idx = list(range(N)); random.Random(seed).shuffle(idx)
    val_len = int(N * val_ratio)
    val_idx, train_idx = idx[:val_len], idx[val_len:]
    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset   = torch.utils.data.Subset(ds, val_idx)
    return train_subset, val_subset

import numpy as np

def mask_batch(seqs, pad_id, unk_id, mask_frac=0.10):
    """
    Randomly replaces 10o% of non-PAD and non-UNK tokens in each sequence with <unk> in the batch.
    """
    maskable = (seqs != pad_id) & (seqs != unk_id)
    masked_seqs = seqs.clone()
    B, L = seqs.size()
    for i in range(B):
        # Get indices of maskable (non-PAD/UNK) positions
        maskable_idxs = maskable[i].nonzero(as_tuple=True)[0].tolist()
        if len(maskable_idxs) == 0:
            continue
        # Calculate how many positions to mask (at least 1)
        n_mask = max(1, round(len(maskable_idxs) * mask_frac))
        mask_pos = np.random.choice(maskable_idxs, n_mask, replace=False)
        # Set selected positions to unk_id
        masked_seqs[i, mask_pos] = unk_id
    return masked_seqs

def run_epoch(model, loader, loss_fn, opt=None, device="cpu",
              mask_randomly=False, pad_id=None, unk_id=None, mask_frac=0.10, mask_batch_probability=0.5):
    total, total_loss = 0, 0.0
    train = opt is not None
    model.train(train)
    for seqs, mask, rts in loader:
        seqs, mask, rts = seqs.to(device), mask.to(device), rts.to(device)
        do_mask = False
        if train and mask_randomly:
            if random.random() < mask_batch_probability:
                do_mask = True
        if do_mask:
            seqs = mask_batch(seqs, pad_id, unk_id, mask_frac=mask_frac)
        preds = model(seqs, mask)
        loss  = loss_fn(preds, rts)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += rts.numel()
        total_loss += loss.item() * rts.numel()
    return total_loss / total

def compute_metrics(y_pred, y_true):
    """
    Computes comprehensive regression metrics.
    Returns dictionary containing various regression metrics
    """
    metrics = {}

    # Basic error metrics
    errors = y_pred - y_true
    abs_errors = torch.abs(errors)
    squared_errors = errors ** 2

    # First-order error metrics
    metrics['MAE'] = torch.mean(abs_errors).item()  # Mean Absolute Error (average magnitude of errors)
    metrics['MSE'] = torch.mean(squared_errors).item()  # Mean Squared Error (sensitive to outliers)
    metrics['RMSE'] = torch.sqrt(torch.tensor(metrics['MSE'])).item()   # Root Mean Squared Error (in original units)

    # Additional error metrics
    metrics['Max_Abs_Error'] = torch.max(abs_errors).item()  # Worst-case error
    metrics['Median_Abs_Error'] = torch.median(abs_errors).item()  # Median error (sensitive to outliers)
    metrics['Mean_Abs_Percentage_Error'] = torch.mean(abs_errors / torch.abs(y_true)).item()  # Relative error as percentage

    # Variability metrics
    metrics['Std_Error'] = torch.std(errors).item()  # Standard deviation of errors (consistency of errors)
    metrics['Std_True'] = torch.std(y_true).item()  # Natural variability in true values
    metrics['Std_Pred'] = torch.std(y_pred).item()  # Variability in predictions (should match Std_True)

    # Model performance metrics
    # R-squared: Proportion of variance explained (1 = perfect fit, 0 = no better than mean)
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum(squared_errors)
    metrics['R2'] = (1 - ss_res / ss_tot).item()

    # Explained variance: Proportion of variance explained (like R2 but ignores mean offset bias)
    # Near-identical R2 and Explained_Variance -> no major bias in errors, errors balance out around 0
    metrics['Explained_Variance'] = (1 - torch.var(errors) / torch.var(y_true)).item()

    # Correlation metrics
    # Pearson: Linear correlation (-1 to 1)
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    metrics['Pearson'] = (torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy))).item()

    # Spearman: Monotonic relationship (-1 to 1)
    metrics['Spearman'] = spearmanr(y_pred.cpu().numpy(), y_true.cpu().numpy()).correlation

    return metrics

