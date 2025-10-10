
# PeptideRT-Transformer

A PyTorch implementation for peptide retention time prediction using Transformer-based models. The main focus is on the encoder architecture, which leverages attention pooling over peptide sequence embeddings for accurate RT regression. The project is designed for proteomics machine learning tasks and is GPU-ready.


## Project Structure

```
data/         # Peptide datasets (TXT: <peptide>\t<rt>)
docs/         # Documentation and figures
src/          # Source code (model, dataset, tokenizer, hpo, utils, training script)
train__cysty.ipynb, train__misc_dia.ipynb   # Jupyter notebooks for encoder training
eval__cysty.ipynb, eval__misc_dia.ipynb     # Jupyter notebooks for encoder evaluation
```

> **Note:** Decoder-based training and evaluation are also implemented. See below for command-line usage, and refer to the codebase and notebooks for more details.


## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install all dependencies, run:

```bash
poetry install
```


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


## Jupyter Notebooks

- **train__cysty.ipynb**, **train__misc_dia.ipynb**: Interactive training and exploration for the encoder model.
- **eval__cysty.ipynb**, **eval__misc_dia.ipynb**: Evaluation and visualization for the encoder model.
- Decoder model usage can be adapted similarly.

---