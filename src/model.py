import math
import copy

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



class CrossOnlyDecoderLayer(nn.Module):
    """
    Custom Decoder Layer that performs ONLY Cross-Attention (Queries <-> Memory)
    and FeedForward. It skips Self-Attention among queries.

    Structure (Post-LN style to match TransformerDecoderLayer defaults):
      x = x + dropout(cross_attn(x, memory, memory))
      x = norm1(x)
      x = x + dropout(ff_block(x))
      x = norm2(m)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(), layer_norm_eps=1e-5, batch_first=False):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Determine activation function
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU() # default
        else:
            self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: (T, B, E) if batch_first=False
        memory: (S, B, E)
        """
        # Cross Attention: query=tgt, key=memory, value=memory
        x = tgt

        attn_out, attn_weights = self.cross_attn(query=x, key=memory, value=memory,
                                                 key_padding_mask=memory_key_padding_mask,
                                                 attn_mask=memory_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feed Forward
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x


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
                 dropout: float       = 0.1,
                 disable_self_attn: bool = False):
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
        if not disable_self_attn:
            decoder_layer = nn.TransformerDecoderLayer(d_model,
                                                       n_heads,
                                                       d_ff,
                                                       dropout,
                                                       batch_first=False) # expects (seq, batch, d)
        else:
            decoder_layer = CrossOnlyDecoderLayer(d_model,
                                                  n_heads,
                                                  dim_feedforward=d_ff,
                                                  dropout=dropout,
                                                  batch_first=False)

        # We manually stack layers instead of using nn.TransformerDecoder
        # because nn.TransformerDecoder expects layers to have specific attributes (e.g. self_attn)
        # which our CrossOnlyDecoderLayer might not have.
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])

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


        out = tgt
        for layer in self.layers:
            # We only pass arguments supported by both TransformerDecoderLayer and CrossOnlyDecoderLayer
            # CrossOnlyDecoderLayer.forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
            # Standard TransformerDecoderLayer also supports these.
            out = layer(out, memory, memory_key_padding_mask=key_padding_mask)

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