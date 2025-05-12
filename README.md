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

The main training script is `src/rt_transformer.py`. It trains the model on a peptide RT dataset and saves the weights to either `rt_encoder_model.pt` or `rt_decoder_model.pt`, depending on the chosen architecture.

**Example usage:**
```bash
python src/rt_transformer.py --data data/mouse.txt --epochs 20 --d_model 256 --layers 10 --arch encoder
python src/rt_transformer.py --data data/mouse.txt --epochs 20 --d_model 256 --layers 10 --arch decoder --queries 4
```
- Use `--arch encoder` or `--arch decoder` to select the model architecture.
- Use `--data` to specify your dataset (TXT file: `<peptide>\t<rt>`).
- Other arguments control hyperparameters such as model size, number of layers, number of queries, learning rate, and more.

## Jupyter Notebooks

- **PeptideRT_Encoder_Train.ipynb**: Interactive training and exploration for the encoder model.
- **PeptideRT_Encoder_Eval.ipynb**: Evaluation and visualization for the encoder model.
- **PeptideRT_Decoder_Train.ipynb**: Interactive training and exploration for the decoder model.
- **PeptideRT_Decoder_Eval.ipynb**: Evaluation and visualization for the decoder model.

---