from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizer import AATokenizer


class PeptideRTDataset(Dataset):
    """
    Dataset for peptide RT data.
    Each line: <peptide sequence>\t<retention time>
    """
    def __init__(self, txt_path: str, tokenizer: AATokenizer):
        self.samples: List[Tuple[torch.LongTensor, float]] = []
        self.tok = tokenizer
        with open(txt_path) as f:
            for line in f:
                if not line.strip():
                    continue
                seq, rt = line.strip().split()
                # Encode sequence to tensor, store with RT as float
                self.samples.append((self.tok.encode(seq), float(rt)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

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