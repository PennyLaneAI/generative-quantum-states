import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.utils import clones


class LayerNorm(nn.Module):
    """ this module implements layer normalization """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """ this module implements a residual connection followed by a layer norm """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm_layer = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x)))


def attention(query, key, value, mask=None, dropout=None):
    """ this function computes the scaled dot product attention mechanism """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """ module implements multihead attention mechanism. Here, we always set dk = dv. """

    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all heads
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        # 1) Do all linear projections in the batch from d_model -> n_heads x d_k
        query, key, value = [
            lin(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) apply attention to all projected vectors in the batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "concatenate" using a view and apply a final linear projection
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x) * math.sqrt(self.d_model)
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # compute the positional encodings once in log space
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        pe = torch.zeros(max_len, d_model // 2)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, dim_1: int, dim_2: int):
        # Compute the positional encodings along two dimensions, then concatenate them
        N, D = x.size(0), x.size(1)
        assert N == dim_1 * dim_2
        pe_1 = self.pe[:dim_1, :D]
        pe_2 = self.pe[:dim_2, :D]
        pe_1 = pe_1.unsqueeze(1).repeat(1, dim_2, 1).view(-1, D)
        pe_2 = pe_2.unsqueeze(0).repeat(dim_1, 1, 1).view(-1, D)
        pe = torch.cat([pe_1, pe_2], dim=1)
        out = x + pe
        return self.dropout(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
