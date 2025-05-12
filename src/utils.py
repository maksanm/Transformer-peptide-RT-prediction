import random

import torch
from torch.utils.data import Dataset

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
        loss  = loss_fn(preds, rts)       # compute MSE loss
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += rts.numel()
        total_loss += loss.item() * rts.numel()
    return total_loss / total

def compute_metrics(y_pred, y_true):
    """
    Copmputes the most popular metrics.
    """
    # y_pred, y_true: torch tensors, shape (N,)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    # R2
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    # Pearson
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    pearson = torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy))
    # Spearman (convert to numpy)
    spearman = spearmanr(y_pred.cpu().numpy(), y_true.cpu().numpy()).correlation
    return dict(MSE=mse, MAE=mae, R2=r2.item(), Pearson=pearson.item(), Spearman=spearman)
