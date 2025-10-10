from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizer import AATokenizer


class PeptideRTDataset(Dataset):
    """
    Dataset for peptide RT data.
    Each line: <peptide sequence>\t<retention time>
    Applies min-max normalization to RTs, but leaves already-normalized data unchanged.
    """
    def __init__(self, txt_path: str, tokenizer: AATokenizer):
        self.samples: List[Tuple[torch.LongTensor, float]] = []
        self.tok = tokenizer
        self.orig_seqs = []
        rts = []
        with open(txt_path) as f:
            for line in f:
                if not line.strip():
                    continue
                seq, rt = line.strip().split()
                self.orig_seqs.append(seq)
                rts.append(float(rt))
        min_rt, max_rt = min(rts), max(rts)
        self.norm_stats = {"min": min_rt, "max": max_rt}
        norm_rts = [
            min(max((rt - min_rt) / (max_rt - min_rt), 0.0), 1.0) if max_rt > min_rt else 0.0
            for rt in rts
        ]
        for seq, norm_rt in zip(self.orig_seqs, norm_rts):
            self.samples.append((self.tok.encode(seq), norm_rt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def decode(self, encoded_seq: torch.LongTensor):
        return self.tok.decode(encoded_seq)

    def normalize(self, y):
        min_rt, max_rt = self.norm_stats["min"], self.norm_stats["max"]
        if min_rt == 0.0 and max_rt == 1.0:
            return y
        norm_y = (y - min_rt) / (max_rt - min_rt) if max_rt > min_rt else 0.0
        return min(max(norm_y, 0.0), 1.0)

    def unnormalize(self, y):
        min_rt, max_rt = self.norm_stats["min"], self.norm_stats["max"]
        if min_rt == 0.0 and max_rt == 1.0:
            return y
        return y * (max_rt - min_rt) + min_rt

def collate(batch, pad_id):
    """
    Collate function for DataLoader.
    Pads sequences in batch to same length, creates key_padding_mask.
    """
    seqs, rts = zip(*batch)                     # list[Tensor], list[float]
    seqs = pad_sequence(seqs, batch_first=True,
                        padding_value=pad_id)   # (B, L) tensor, padded
    rts  = torch.tensor(rts, dtype=torch.float) # (B,) tensor of RTs
    # attention mask: True for PAD (that is what nn.Transformer expects)
    key_padding_mask = seqs.eq(pad_id)          # (B,L) -> bool mask
    return seqs, key_padding_mask, rts


'''
a = AATokenizer()
d = PeptideRTDataset("data/mouse.txt", a)

i = [d.__getitem__(id) for id in range(100, 110)]

print(i)

print(collate(i, a.pad_id))
'''