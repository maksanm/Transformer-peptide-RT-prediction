# PeptideRT-Transformer

A PyTorch implementation of a peptide retention time predictor using a Transformer decoder with multiple learnable query tokens and attention pooling before the regression head. The model leverages cross-attention from learned queries to peptide sequence embeddings for accurate RT prediction. Configurable and GPU-ready for proteomics machine learning tasks.

## Project Structure

```
data/         # Peptide datasets (TXT: <peptide>\t<rt>)
docs/         # Documentation
src/          # Source code (model, dataset, tokenizer, utils, main script)
PeptideRT_Train.ipynb   # Jupyter notebook for training
PeptideRT_Eval.ipynb    # Jupyter notebook for evaluation
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install all dependencies, run:

```bash
poetry install
```

## How to Train

The main training script is `src/rt_transformer.py`.
It trains the model on a peptide RT dataset and saves the weights to `rt_model.pt`.

**Example usage:**
```bash
poetry run python src/rt_transformer.py --data data/mouse.txt --epochs 20 --d_model 256 --layers 10 --queries 4
```
- Use `--data` to specify your dataset (TXT file: `<peptide>\t<rt>`).
- Other arguments control model size, number of layers, queries, etc.

## Jupyter Notebooks

- **PeptideRT_Train.ipynb**: Interactive training and exploration.
- **PeptideRT_Eval.ipynb**: Model evaluation and visualization.
