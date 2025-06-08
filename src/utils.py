import random

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple

from scipy.stats import spearmanr


def split_dataset(ds: Dataset, val_ratio=0.1, seed=42):
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

def run_epoch(model, loader, loss_fn, opt=None, device="cpu"):
    """
    Run one epoch (train if opt is given, else eval).
    Returns average loss.
    """
    total, total_loss = 0, 0.0
    train = opt is not None
    model.train(train)  # set train/eval mode
    for seqs, mask, rts in loader:
        seqs, mask, rts = seqs.to(device), mask.to(device), rts.to(device)
        preds = model(seqs, mask)         # forward pass
        loss  = loss_fn(preds, rts)       # compute loss
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

