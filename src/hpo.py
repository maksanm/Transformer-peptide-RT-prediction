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
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import HyperbandPruner

try:
    from optuna.integration import BoTorchSampler
    HAS_BOTORCH = True
except Exception:
    HAS_BOTORCH = False

from dataset import PeptideRTDataset, collate
from functools import partial

from model import PeptideRTEncoderModel
from tokenizer import AATokenizer
from utils import run_epoch, split_dataset, compute_metrics


DATASET="misc_dia"
DATA = f"data/{DATASET}.txt"
OUTPUT_DIR = f"models/hpo/{DATASET}/"
SEED = 42

# HPO settings
# Using 150 total trials: with 315 total configs (7×9×5) and ~14% invalid (45 configs where d_model % n_head != 0),
# we expect ≈129 valid evaluations - enough coverage given small search space
N_TRIALS = 150                # total HPO trials
MAX_EPOCHS = 100              # max epochs per trial
PATIENCE = 20                 # early stop inside a trial if no val improvement
N_JOBS = 4                    # number of Optuna parallel jobs (processes); set >1 to parallelize trials
PIN_MEMORY = True

# Search space
D_MODEL_CHOICES = [64, 96, 128, 160, 192, 256, 320]
LAYER_CHOICES   = [1, 2, 3, 4, 5, 6, 7, 8, 10]
HEAD_CHOICES    = [1, 2, 4, 6, 8]

# Fixed hyperparameters
BATCH_SIZE     = 256            # fixed batch size
LEARNING_RATE  = 1e-3          # fixed learning rate


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(tokenizer, hparams, device):
    """Build model on a specific device.

    If multiple GPUs are available and we are running only one Optuna job (N_JOBS == 1),
    we wrap the model in DataParallel to use all GPUs for a single trial. If we are
    running multiple Optuna jobs in parallel (N_JOBS > 1), each trial should occupy
    a single GPU to avoid resource contention, so we DO NOT wrap in DataParallel.
    """
    model = PeptideRTEncoderModel(
        tokenizer,
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        d_ff=4 * hparams["d_model"],
        n_layers=hparams["n_layers"],
    )
    use_dp = torch.cuda.device_count() > 1 and N_JOBS == 1 and device.type == "cuda"
    model = model.to(device)
    if use_dp:
        print(f"Wrapping model in DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    return model


def get_trial_device(trial_number: int) -> torch.device:
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return torch.device("cpu")
    if N_JOBS == 1:
        return torch.device("cuda:0")
    # Deterministic assignment for first N_JOBS trials
    if trial_number < N_JOBS:
        return torch.device(f"cuda:{trial_number % gpu_count}")
    # Otherwise, use nvidia-smi to pick least-loaded GPU
    import subprocess, re
    try:
        time.sleep(2)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, check=True
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            idx, util, mem_used, mem_total = [int(x.strip()) for x in line.split(",")]
            mem_ratio = mem_used / (mem_total + 1e-6)
            candidates.append((idx, util, mem_ratio))
        if not candidates:
            return torch.device("cuda:0")
        # choose GPU with lowest util, then lowest memory ratio as tiebreaker
        idx, _, _ = min(candidates, key=lambda x: (x[1], x[2]))
        return torch.device(f"cuda:{idx}")
    except Exception:
        return torch.device("cuda:0")


def objective_factory(tokenizer, train_ds, val_ds, checkpoint_dir):
    def objective(trial: optuna.trial.Trial):
        n_layers = trial.suggest_categorical("n_layers", LAYER_CHOICES)
        n_heads = trial.suggest_categorical("n_heads", HEAD_CHOICES)
        d_model = trial.suggest_categorical("d_model", D_MODEL_CHOICES)

        if d_model % n_heads != 0:
            raise optuna.TrialPruned("d_model is not divisible by n_heads")

        batch = BATCH_SIZE
        lr = LEARNING_RATE

        hparams = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "lr": lr,
            "batch": batch,
        }

        # Assign a (possibly different) GPU for this trial when running multi-job HPO
        trial_device = get_trial_device(trial.number)
        if trial_device.type == "cuda":
            torch.cuda.set_device(trial_device)

        print(f"\n[trial {trial.number:02d}] running on device {trial_device} using config: "
              f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")

        collate_fn = partial(collate, pad_id=tokenizer.pad_id)
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
            num_workers=0, pin_memory=(PIN_MEMORY and trial_device.type == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn,
            num_workers=0, pin_memory=(PIN_MEMORY and trial_device.type == "cuda")
        )

        model = build_model(tokenizer, hparams, trial_device)
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
                tr_loss = run_epoch(model, train_loader, loss_fn, opt, trial_device)
                vl_loss = run_epoch(model, val_loader,   loss_fn, None, trial_device)
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
        best_model = build_model(tokenizer, hparams, trial_device)
        state_dict = torch.load(trial_ckpt, map_location=trial_device)
        # Handle potential DataParallel 'module.' prefix differences gracefully
        try:
            best_model.load_state_dict(state_dict)
        except RuntimeError:
            # Strip 'module.' prefixes if needed
            new_state = {k.split('module.', 1)[-1]: v for k, v in state_dict.items()}
            best_model.load_state_dict(new_state)
        best_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for seqs, mask, rts in val_loader:
                seqs, mask, rts = seqs.to(trial_device), mask.to(trial_device), rts.to(trial_device)
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
        if trial_device.type == "cuda":
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"Using device: {device} ({gpu_count} GPUs available)")
    if N_JOBS > gpu_count:
        raise RuntimeError(f"N_JOBS ({N_JOBS}) > available GPUs ({gpu_count}): HPO blocked to prevent resource contention.")

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

    objective = objective_factory(tokenizer, train_ds, val_ds, checkpoint_dir)
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