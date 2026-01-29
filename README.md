
# PeptideRT-Transformer

A PyTorch implementation for peptide retention time prediction using Transformer-based models. The main focus is on the encoder architecture, which leverages attention pooling over peptide sequence embeddings for accurate RT regression. The project is designed for proteomics machine learning tasks and is GPU-ready.


## Project Structure

```
data/         # Peptide datasets (TXT: <peptide>\t<rt>)
docs/         # Documentation and figures
src/          # Source code (model, dataset, tokenizer, hpo, utils, training script)
encoder_notebooks/  # Notebooks for training & evaluating Encoder models
decoder_notebooks/  # Notebooks for training & evaluating Decoder models
```

> **Note:** The `notebooks` have been organized by architecture for clarity.


## Installation

This project uses [Poetry](https://python-poetry.org/) to manage Python dependencies and virtual environments in a robust, reproducible way.

1. **Install Poetry** (if not already installed):
   Follow the instructions at [python-poetry.org/docs/](https://python-poetry.org/docs/)

2. **Install project dependencies**:
   ```bash
   poetry install
   ```
   This will create a virtual environment and install all required packages (PyTorch, Optuna, etc.) specified in `pyproject.toml`.


## How to Train

The main training script is `src/rt_transformer.py`. It requires `--data` and `--output` parameters, and supports various hyperparameters.

**Encoder example:**
```bash
poetry run python src/rt_transformer.py --data data/mouse.txt --epochs 150 --d_model 64 --layers 5 --arch encoder --output encoder_model.pt
```

**Decoder example:**
```bash
poetry run python src/rt_transformer.py --data data/mouse.txt --epochs 150 --d_model 64 --layers 5 --arch decoder --queries 4 --output decoder_model.pt
```

- Use `--arch encoder` (default) or `--arch decoder` to select the model architecture.
- Other arguments: `--batch`, `--heads`, `--lr`, `--seed`, etc. See `src/rt_transformer.py` for all options.

## Hyperparameter Optimization (HPO)

Automated hyperparameter tuning is available via separate scripts for each architecture. These use **Optuna** (with optional **BoTorch** sampling) to find optimal configurations.

- **Encoder HPO**:
  ```bash
  poetry run python src/hpo_encoder.py
  ```
  *(optimizes: d_model, n_layers, n_heads)*

- **Decoder HPO**:
  ```bash
  poetry run python src/hpo_decoder.py
  ```
  *(optimizes: d_model, n_layers, n_heads, n_queries, disable_self_attn)*

*Note: Edit the configuration constants (e.g., `DATASET`, `N_TRIALS`) directly at the top of these Python scripts before running.*


## Jupyter Notebooks

Results, visualizations, and interactive training logs are available in the notebook directories:

- **`encoder_notebooks/`**: Contains `train__*.ipynb` and `eval__*.ipynb` for the Encoder architecture.
- **`decoder_notebooks/`**: Contains `train__*.ipynb` and `eval__*.ipynb` for the Decoder architecture (including masked training variants).

To launch the notebook server within the poetry environment:
```bash
poetry run jupyter notebook
```
