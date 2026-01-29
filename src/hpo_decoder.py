"""
HPO for Transformer decoder on peptide RT prediction.
- Optimizes: d_model, n_layers, n_heads, n_queries, disable_self_attn
- Decoder-only (with optional self-attention disabled)
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

from model import PeptideRTDecoderModel
from tokenizer import AATokenizer
from utils import run_epoch, split_dataset, compute_metrics


DATASET="cysty"
DATA = f"data/{DATASET}.txt"
OUTPUT_DIR = f"models/hpo_decoder/{DATASET}/"
SEED = 42

# HPO settings
# Using 600 total trials: with 4410 total configs (7×9×5×7×2) and ~14% invalid (where d_model % n_head != 0),
# we expect ≈3780 valid evaluations - ~16% coverage is sufficient for Bayesian optimization
N_TRIALS = 600                # total HPO trials
MAX_EPOCHS = 100              # max epochs per trial
PATIENCE = 20                 # early stop inside a trial if no val improvement
N_JOBS = 4                    # number of Optuna parallel jobs
PIN_MEMORY = True

# Search space
D_MODEL_CHOICES = [64, 96, 128, 160, 192, 256, 320]
LAYER_CHOICES   = [1, 2, 3, 4, 5, 6, 7, 8, 10]
HEAD_CHOICES    = [1, 2, 4, 6, 8, 12]

# Decoder specific parameters
N_QUERIES_CHOICES = [4, 8, 12, 16, 20, 24]
DISABLE_SELF_ATTN_CHOICES = [True, False]

# Fixed hyperparameters
BATCH_SIZE     = 512            # fixed batch size
LEARNING_RATE  = 5e-4           # fixed learning rate


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(tokenizer, hparams, device):
    """Build model on a specific device."""
    model = PeptideRTDecoderModel(
        tokenizer,
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        d_ff=4 * hparams["d_model"],
        n_layers=hparams["n_layers"],
        n_queries=hparams["n_queries"],
        disable_self_attn=hparams["disable_self_attn"]
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

        n_queries = trial.suggest_categorical("n_queries", N_QUERIES_CHOICES)
        disable_self_attn = trial.suggest_categorical("disable_self_attn", DISABLE_SELF_ATTN_CHOICES)

        if d_model % n_heads != 0:
            raise optuna.TrialPruned("d_model is not divisible by n_heads")

        batch = BATCH_SIZE
        lr = LEARNING_RATE

        hparams = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_queries": n_queries,
            "disable_self_attn": disable_self_attn,
            "lr": lr,
            "batch": batch,
        }

        # Assign a (possibly different) GPU for this trial when running multi-job HPO
        trial_device = get_trial_device(trial.number)
        if trial_device.type == "cuda":
            torch.cuda.set_device(trial_device)

        print(f"\n[trial {trial.number:02d}] running on device {trial_device} using config: "
              f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, "
              f"n_queries={n_queries}, disable_self_attn={disable_self_attn}")

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

        trial_ckpt = os.path.join(checkpoint_dir, f"trial_{trial.number}.pt")
        trial_cfg  = os.path.join(checkpoint_dir, f"trial_{trial.number}.json")

        try:
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

                # Report intermediate value to Optuna
                trial.report(vl_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
        except optuna.TrialPruned:
            # Cleanup checkpoint for pruned trials
            if os.path.exists(trial_ckpt):
                try:
                    os.remove(trial_ckpt)
                except OSError: pass
            if os.path.exists(trial_cfg):
                try:
                    os.remove(trial_cfg)
                except OSError: pass
            raise

        try:
            global_best = trial.study.best_value
        except ValueError:
            global_best = float("inf")

        if float(best_val) >= global_best:
            if os.path.exists(trial_ckpt):
                try:
                    os.remove(trial_ckpt)
                except OSError: pass
            if os.path.exists(trial_cfg):
                try:
                    os.remove(trial_cfg)
                except OSError: pass
        else:
            print(f"Trial {trial.number} is new global best ({float(best_val):.6f} < {global_best:.6f}). Keeping checkpoint.")

        return best_val

    return objective


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Loading data from {DATA}...")
    tokenizer = AATokenizer()
    ds = PeptideRTDataset(DATA, tokenizer)
    train_ds, val_ds = split_dataset(ds)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    print("Setting up HPO...")

    # Check if we should use Multi-GPU via Optuna jobs (parallelism)
    # If N_JOBS > 1 and we have multiple GPUs, each job runs on one GPU (assigned in objective).

    # Sampler setup
    if HAS_BOTORCH:
        print("Using BoTorchSampler (GP-based bayesian opt).")
        sampler = BoTorchSampler(n_startup_trials=20)
    else:
        print("BoTorch not found, falling back to TPESampler.")
        sampler = optuna.samplers.TPESampler(n_startup_trials=20, seed=SEED)

    pruner = HyperbandPruner(min_resource=5, max_resource=MAX_EPOCHS, reduction_factor=3)

    study_path = f"sqlite:///{OUTPUT_DIR}/hpo.db"
    study_name = f"peptide_rt_decoder_{DATASET}"

    try:
        optuna.delete_study(study_name=study_name, storage=study_path)
    except Exception:
        pass

    study = optuna.create_study(
        study_name=study_name,
        storage=study_path,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    print(f"Starting optimization with {N_TRIALS} trials...")

    obj_func = objective_factory(tokenizer, train_ds, val_ds, checkpoint_dir)

    try:
        study.optimize(obj_func, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        print("HPO interrupted by user.")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best config separately
    best_config_path = os.path.join(OUTPUT_DIR, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(trial.params, f, indent=4)

    # Copy best model
    best_trial_pt = os.path.join(checkpoint_dir, f"trial_{trial.number}.pt")
    best_model_pt = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(best_trial_pt):
        shutil.copy(best_trial_pt, best_model_pt)
        print(f"Best model saved to {best_model_pt}")
    else:
        print("Best model checkpoint not found (maybe pruned?)")

if __name__ == "__main__":
    main()
