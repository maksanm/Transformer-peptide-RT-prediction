# PeptideRT-Transformer

A PyTorch implementation of a peptide retention time predictor using Transformer encoder or decoder architectures. The encoder model uses attention pooling over sequence embeddings, while the decoder model supports multiple learnable query tokens with cross-attention to peptide embeddings before the regression head. Both are configurable and GPU-ready for proteomics machine learning tasks.

## Project Structure

```
data/         # Peptide datasets (TXT: <peptide>\t<rt>)
docs/         # Documentation
src/          # Source code (model, dataset, tokenizer, utils, main script)
PeptideRT_Encoder_Train.ipynb    # Jupyter notebook for encoder training
PeptideRT_Encoder_Eval.ipynb     # Jupyter notebook for encoder evaluation
PeptideRT_Decoder_Train.ipynb    # Jupyter notebook for decoder training
PeptideRT_Decoder_Eval.ipynb     # Jupyter notebook for decoder evaluation
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install all dependencies, run:

```bash
poetry install
```

## How to Train

The main training script, `src/rt_transformer.py`, requires `--data` and `--output` parameters: it trains the model on the specified peptide retention time dataset and saves the model weights to the given output path.

**Example usage:**
```bash
poetry run python src/rt_transformer.py --data data/mouse.txt --epochs 150 --d_model 64 --layers 5 --arch encoder --output encoder_model.pt
poetry run python src/rt_transformer.py --data data/mouse.txt --epochs 150 --d_model 64 --layers 5 --arch decoder --queries 4 --output decoder_model.pt
```
- Use `--arch encoder` or `--arch decoder` to select the model architecture.
- Other arguments control hyperparameters such as model size, number of layers, number of queries, learning rate, and more.

## Jupyter Notebooks

- **PeptideRT_Encoder_Train.ipynb**: Interactive training and exploration for the encoder model.
- **PeptideRT_Encoder_Eval.ipynb**: Evaluation and visualization for the encoder model.
- **PeptideRT_Decoder_Train.ipynb**: Interactive training and exploration for the decoder model.
- **PeptideRT_Decoder_Eval.ipynb**: Evaluation and visualization for the decoder model.

---