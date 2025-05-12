from typing import List
import torch


class AATokenizer:
    """
    Tokenizer for amino acids.
    20 natural AA + few rare ones + PAD + UNK  ==> integer ids
    """
    def __init__(self):
        aas = list("ACDEFGHIKLMNPQRSTVWY")          # 20 canonical amino acids
        extras = list("BJOUZ")                     # non-standard amino acids: B (N/D), J (rare), O (part of selenocysteine), U (selenocysteine), Z (Q/E)
        self.pad_token, self.unk_token = "<pad>", "<unk>"
        vocab = [self.pad_token, self.unk_token] + aas + extras
        self.stoi = {t: i for i, t in enumerate(vocab)}  # string to index
        self.itos = {i: t for t, i in self.stoi.items()} # index to string
        self.pad_id = self.stoi[self.pad_token]          # index for padding
        self.unk_id = self.stoi[self.unk_token]          # index for unknown

    def encode(self, seq: str) -> torch.LongTensor:
        # Convert each character to its index, use unk_id if not found
        ids = [self.stoi.get(ch, self.unk_id) for ch in seq.strip()]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        # Convert indices back to string, skip padding
        return "".join(self.itos[i] for i in ids if i not in (self.pad_id,))

    @property
    def vocab_size(self):
        return len(self.stoi)


'''
a = AATokenizer()

print('-----------')

t=a.encode('FGHIKEAFDPHASXCMZXXC]')
print(t)

print(a.decode(t.tolist()))
'''