"""
HPO for Transformer encoder on peptide RT prediction using Bayesian optimization + Hyperband pruning.
- No CLI args. Configure constants below.
- Optimizes: d_model, n_layers, n_heads
- Encoder-only
- Fast-converging setup: Optuna + (BoTorch GP/EI sampler if available, else TPE) + Hyperband pruner
"""

import os
import json
import time
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import HyperbandPruner

try:
    # Optional: GP-based, highly sample-efficient sampler
    from optuna.integration import BoTorchSampler
    HAS_BOTORCH = True
except Exception:
    HAS_BOTORCH = False

from dataset import PeptideRTDataset, collate
from functools import partial

from model import PeptideRTEncoderModel
from tokenizer import AATokenizer
from utils import run_epoch, split_dataset, compute_metrics


DATASET="cysty"
DATA = f"data/{DATASET}.txt"
OUTPUT_DIR = f"models/hpo/{DATASET}/"
SEED = 42

# HPO settings
N_TRIALS = 150                # total HPO trials
MAX_EPOCHS = 100              # max epochs per trial
PATIENCE = 20                 # early stop inside a trial if no val improvement
N_JOBS = 1                    # number of Optuna parallel jobs (processes)
PIN_MEMORY = True

# Search space
D_MODEL_CHOICES = [32, 64, 96, 128, 160, 192, 256]
LAYER_CHOICES   = [1, 2, 3, 4, 5, 6, 7, 8, 10]
HEAD_CHOICES    = [1, 2, 4, 6, 8]

# Fixed hyperparameters
BATCH_SIZE     = 64            # fixed batch size
LEARNING_RATE  = 3e-4          # fixed learning rate


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(tokenizer, hparams, device):
    model = PeptideRTEncoderModel(
        tokenizer,
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        d_ff=4*hparams["d_model"],
        n_layers=hparams["n_layers"]
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model


def objective_factory(tokenizer, train_ds, val_ds, device, checkpoint_dir):
    def objective(trial: optuna.trial.Trial):
        n_heads = trial.suggest_categorical("n_heads", HEAD_CHOICES)
        d_model = trial.suggest_categorical("d_model", D_MODEL_CHOICES)
        n_layers = trial.suggest_categorical("n_layers", LAYER_CHOICES)
        batch = BATCH_SIZE
        lr = LEARNING_RATE

        # Early prune if d_model is not divisible by n_heads
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()

        hparams = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "lr": lr,
            "batch": batch,
        }

        collate_fn = partial(collate, pad_id=tokenizer.pad_id)
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
            pin_memory=(PIN_MEMORY and device == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn,
            pin_memory=(PIN_MEMORY and device == "cuda")
        )

        model = build_model(tokenizer, hparams, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()

        best_val = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        # Per-trial checkpoint path (we'll save the best of this trial here)
        trial_ckpt = os.path.join(checkpoint_dir, f"trial_{trial.number}.pt")
        trial_cfg  = os.path.join(checkpoint_dir, f"trial_{trial.number}.json")

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()
            try:
                tr_loss = run_epoch(model, train_loader, loss_fn, opt, device)
                vl_loss = run_epoch(model, val_loader,   loss_fn, None, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raise optuna.TrialPruned()  # treat OOM as pruned
                else:
                    raise

            improved = vl_loss < best_val - 1e-8
            if improved:
                best_val = vl_loss
                best_epoch = epoch
                # Save best checkpoint for this trial
                torch.save(model.state_dict(), trial_ckpt)
                with open(trial_cfg, "w") as f:
                    json.dump({
                        "best_val": float(best_val),
                        "best_epoch": best_epoch,
                        **hparams,
                        "seed": SEED
                    }, f, indent=2)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Report to Optuna for pruning
            trial.report(vl_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Local early stopping to cut wasted compute
            if epochs_no_improve >= PATIENCE:
                break

            dt = time.time() - t0
            print(f"[trial {trial.number:02d}] epoch {epoch:03d} | "
                  f"train {tr_loss:.4e} | val {vl_loss:.4e} | "
                  f"best {best_val:.4e} @ {best_epoch:03d} | {dt:.1f}s")

        # After the trial, print statistics for the best epoch (best checkpoint)
        # Load best model
        best_model = build_model(tokenizer, hparams, device)
        best_model.load_state_dict(torch.load(trial_ckpt, map_location=device))
        best_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for seqs, mask, rts in val_loader:
                seqs, mask, rts = seqs.to(device), mask.to(device), rts.to(device)
                preds = best_model(seqs, mask)
                all_preds.append(preds.cpu())
                all_targets.append(rts.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        stats = compute_metrics(all_preds, all_targets)
        print(f"[trial {trial.number:02d}] Best epoch {best_epoch:03d} | Validation statistics:")
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}")

        # Return the best validation loss obtained in this trial
        # (Optuna will keep the best over trials)
        del model
        torch.cuda.empty_cache()
        return float(best_val)

    return objective


def main():
    if HAS_BOTORCH:
        print("Using BoTorchSampler.")
    else:
        print("BoTorchSampler not available, falling back to TPESampler.")
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not Path(DATA).exists():
        raise FileNotFoundError(f"DATA not found: {DATA}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Prepare data once, keep split fixed across all trials
    tokenizer = AATokenizer()
    ds = PeptideRTDataset(DATA, tokenizer)
    train_ds, val_ds = split_dataset(ds, val_ratio=0.05, seed=SEED)

    # Optuna study: use a sample-efficient sampler and a multi-fidelity pruner
    sampler = None
    if HAS_BOTORCH:
        # GP/EI sampler is very sample-efficient for low dimension (we have 5)
        sampler = BoTorchSampler(seed=SEED)
    else:
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8, multivariate=True)

    pruner = HyperbandPruner(min_resource=5, max_resource=MAX_EPOCHS, reduction_factor=3)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="encoder_hpo")

    objective = objective_factory(tokenizer, train_ds, val_ds, device, checkpoint_dir)
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, n_jobs=N_JOBS)

    # Persist best artifacts
    best_trial = study.best_trial
    best_ckpt = os.path.join(checkpoint_dir, f"trial_{best_trial.number}.pt")
    best_cfg  = os.path.join(checkpoint_dir, f"trial_{best_trial.number}.json")

    final_ckpt = os.path.join(OUTPUT_DIR, "best_encoder.pt")
    final_cfg  = os.path.join(OUTPUT_DIR, "best_config.json")
    if Path(best_ckpt).exists():
        shutil.copyfile(best_ckpt, final_ckpt)
    if Path(best_cfg).exists():
        shutil.copyfile(best_cfg, final_cfg)

    print("\n=== HPO finished ===")
    print(f"Best trial #{best_trial.number} | val loss = {best_trial.value:.4e}")
    print(f"Best model: {final_ckpt}")
    print(f"Best config: {final_cfg}")
    print("✔ done.")


if __name__ == "__main__":
    main()