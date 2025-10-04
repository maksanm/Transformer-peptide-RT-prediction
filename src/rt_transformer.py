"""
Transformer for peptide retention-time prediction.
"""

import argparse, random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PeptideRTDataset, collate
from model import PeptideRTEncoderModel, PeptideRTDecoderModel
from tokenizer import AATokenizer
from utils import run_epoch, split_dataset, compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="TXT file:  <peptide>\\t<rt>")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--heads",   type=int, default=4)
    parser.add_argument("--layers",  type=int, default=8)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--arch", choices=["encoder", "decoder"], default="encoder",
                        help="Model architecture: encoder or decoder")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AATokenizer()
    ds  = PeptideRTDataset(args.data, tokenizer)
    train_ds, val_ds = split_dataset(ds)
    coll = lambda b: collate(b, tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, collate_fn=coll)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=coll)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.arch == "encoder":
        model = PeptideRTEncoderModel(
            tokenizer,
            d_model=args.d_model,
            n_heads=args.heads,
            d_ff=4*args.d_model,
            n_layers=args.layers
        )
    else:
        model = PeptideRTDecoderModel(
            tokenizer,
            d_model=args.d_model,
            n_heads=args.heads,
            d_ff=4*args.d_model,
            n_layers=args.layers,
            n_queries=args.queries
        )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs+1):
        tr_loss = run_epoch(model, train_loader, loss_fn, opt, device)
        vl_loss = run_epoch(model, val_loader,   loss_fn, None, device)
        print(f"Epoch {epoch:3d} | train loss {tr_loss:.4e} | val loss {vl_loss:.4e}")
        torch.save(model.state_dict(), args.output)
    print("✔ done.")

if __name__ == "__main__":
    main()