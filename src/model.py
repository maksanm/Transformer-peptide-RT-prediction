import math

import torch
import torch.nn as nn

from tokenizer import AATokenizer


def sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding (as in Vaswani et al. 2017).
    Returns (max_len, d_model) tensor.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))              # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term)                         # even indices
    pe[:, 1::2] = torch.cos(position * div_term)                         # odd indices
    return pe                                       # (max_len, d_model)


# ---------------------------  encoder model  ---------------------------------------- #
class PeptideRTEncoderModel(nn.Module):
    """
    Transformer encoder-based model for peptide retention time prediction.

    - Embeds the input peptide sequence with positional encoding.
    - Processes the sequence with a stack of Transformer encoder layers.
    - Pools the outputs of the encoder using attention pooling.
    - Predicts retention time with a regression head.
    """
    def __init__(self,
                 tokenizer: AATokenizer,
                 d_model: int         = 128,
                 n_heads: int         = 8,
                 d_ff: int            = 4*128,
                 n_layers: int        = 4,
                 dropout: float       = 0.1):
        super().__init__()

        # Store tokenizer for dynamic masking during training
        self.tokenizer = tokenizer

        # Embedding for amino acid tokens (with padding)
        self.embed = nn.Embedding(tokenizer.vocab_size, d_model, tokenizer.pad_id)
        # Register a fixed (non-trainable) sinusoidal PE as buffer
        max_len = 512
        self.register_buffer("pos_enc",
                             sinusoidal_position_encoding(max_len, d_model),
                             persistent=False)

        # Transformer encoder: stack of identical layers
        encoder_layer = nn.TransformerEncoderLayer(d_model,
                                                   n_heads,
                                                   d_ff,
                                                   dropout,
                                                   batch_first=True)  # (B, L, d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=n_layers)

        # Attention pooling over sequence positions
        self.attn_pool = nn.Linear(d_model, 1)  # (d_model) -> (1), shared for all positions

        # Regression head: expects d_model input (after pooling)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.d_model = d_model
        self.max_len = max_len

    def forward(self,
                seq_ids: torch.LongTensor,               # (B, L)
                key_padding_mask: torch.BoolTensor):     # (B, L)
        B, L = seq_ids.size()
        if L > self.max_len:
            raise RuntimeError(f"Sequence longer ({L}) than max_len ({self.max_len})")

        # Embedding lookup + add positional encoding
        # emb: (B, L, d_model)
        emb = self.embed(seq_ids) + self.pos_enc[:L].unsqueeze(0)  # (1, L, d_model)

        # Run encoder: (B, L, d_model)
        out = self.encoder(emb, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        # Attention pooling over sequence positions
        # Compute attention logits for each position (B, L, 1)
        attn_logits = self.attn_pool(out)  # (B, L, 1)
        # Mask out PAD positions by setting logits to large negative value
        attn_logits = attn_logits.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=1)  # softmax over sequence positions (L)
        # Weighted sum over sequence positions
        out = (out * attn_weights).sum(dim=1)  # (B, d_model)

        # Regression head: map to scalar
        pred = self.reg_head(out).squeeze(-1) # (B,)
        return pred


# ---------------------------  decoder model  ---------------------------------------- #
class PeptideRTDecoderModel(nn.Module):
    """
    Transformer decoder-based model for peptide retention time prediction.

    - Embeds the input peptide sequence with positional encoding.
    - Uses multiple learnable query vectors that attend to the peptide sequence via a Transformer decoder.
    - Pools the outputs of the queries using attention pooling.
    - Predicts retention time with a regression head.
    """
    def __init__(self,
                 tokenizer: AATokenizer,
                 d_model: int         = 128,
                 n_heads: int         = 8,
                 d_ff: int            = 4*128,
                 n_layers: int        = 4,
                 n_queries: int       = 4,
                 dropout: float       = 0.1):
        super().__init__()

        # Store tokenizer for dynamic masking during training
        self.tokenizer = tokenizer

        # Embedding for amino acid tokens (with padding)
        self.embed = nn.Embedding(tokenizer.vocab_size, d_model, tokenizer.pad_id)
        # Register a fixed (non-trainable) sinusoidal PE as buffer
        max_len = 512
        self.register_buffer("pos_enc",
                             sinusoidal_position_encoding(max_len, d_model),
                             persistent=False)

        # Multiple learned query tokens (shape: n_queries, 1, d_model)
        self.queries = nn.Parameter(torch.randn(n_queries, 1, d_model))

        # Transformer decoder: stack of identical layers
        decoder_layer = nn.TransformerDecoderLayer(d_model,
                                                   n_heads,
                                                   d_ff,
                                                   dropout,
                                                   batch_first=False) # expects (seq, batch, d)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=n_layers)

        # Attention pooling over queries
        self.query_attn = nn.Linear(d_model, 1)  # (d_model) -> (1), shared for all queries

        # Regression head: expects d_model input (after pooling)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.d_model = d_model
        self.max_len = max_len
        self.n_queries = n_queries

    def forward(self,
                seq_ids: torch.LongTensor,               # (B, L)
                key_padding_mask: torch.BoolTensor):     # (B, L)
        B, L = seq_ids.size()
        if L > self.max_len:
            raise RuntimeError(f"Sequence longer ({L}) than max_len ({self.max_len})")

        # Embedding lookup + add positional encoding
        # emb: (B, L, d_model)
        emb = self.embed(seq_ids) + self.pos_enc[:L].unsqueeze(0)  # (1, L, d_model) broadcast
        # Transformer expects (seq_len, batch, d_model)
        memory = emb.transpose(0, 1)                     # (L, B, d_model)

        # Decoder input: learned queries, repeated for each batch element
        # tgt: (n_queries, B, d_model)
        tgt = self.queries.repeat(1, B, 1)               # (n_queries, B, d_model)

        # Run decoder: cross-attends queries to memory (peptide sequence)
        # memory_key_padding_mask: (B, L), True for PAD
        out = self.decoder(tgt,
                           memory,
                           memory_key_padding_mask=key_padding_mask)
        # out: (n_queries, B, d_model)

        # Attention pooling over queries
        # Compute attention logits for each query (n_queries, B, 1)
        attn_logits = self.query_attn(out)  # (n_queries, B, 1)
        attn_weights = torch.softmax(attn_logits, dim=0)  # softmax over queries (dim=0)
        # Weighted sum over queries
        out = (out * attn_weights).sum(dim=0)  # (B, d_model)

        # Regression head: map to scalar
        pred = self.reg_head(out).squeeze(-1)            # (B,)
        return pred