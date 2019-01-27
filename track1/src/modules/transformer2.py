import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, dropout_rate=0.2):
        super(MultiHeadAttention, self).__init__()

        if dim_out % n_heads != 0:
            raise ValueError(
                'MultiHeadSelfAttention: Output dimensions {dim_out} isn\'t'
                'a multiplier of number of heads {n_heads}'.format(
                    dim_out=dim_out,
                    n_heads=n_heads
                )
            )

        self.n_heads = n_heads
        self.dim_head = dim_out // n_heads
        self.linear_q = nn.Linear(dim_in, dim_out)
        self.linear_k = nn.Linear(dim_in, dim_out)
        self.linear_v = nn.Linear(dim_in, dim_out)
        self.linear_o = nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, query, value, mask):
        """
        Args:
            query (FloatTensor): (batch_size, query_len, dim_in)
            value (FloatTensor): (batch_size, value_len, dim_in)
            mask (ByteTensor): (batch_size, value_len)

        Returns:
            (batch_size, query_len, dim_out)
        """
        batch_size, query_len, dim_feature = query.shape
        batch_size, value_len, dim_feature = value.shape

        q = self.linear_q(query)
        k, v = self.linear_k(value), self.linear_v(value)
        q, k, v = [
            x.reshape(batch_size, -1, self.n_heads, self.dim_head)
             .transpose(1, 2)
            for x in (q, k, v)
        ]
        # q.shape == (batch_size, n_heads, query_len, dim_head)
        # k.shape == v.shape == (batch_size, n_heads, value_len, dim_head)

        # calculate attention score
        score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.dim_head)

        # mask out padding and apply attention weights
        score.masked_fill_(
            mask.view(batch_size, 1, 1, value_len) == 0, -math.inf
        )
        weights = F.softmax(score, dim=-1)
        weights = self.dropout(weights)
        # weight.shape == (batch_size, n_heads, query_len, value_len)

        attention = torch.matmul(weights, v)
        # attention.shape == (batch_size, n_heads, query_len, dim_head)

        # reshape to concat results
        attention = attention.transpose(1, 2) \
                             .reshape(batch_size, query_len,
                                      self.n_heads * self.dim_head)
        output = self.linear_o(attention)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, dim_input, dim_output, n_heads, dropout_rate, dim_ff):
        super(EncoderBlock, self).__init__()
        self.self_attn = \
            MultiHeadAttention(dim_input, dim_output, n_heads, dropout_rate)
        self.ff = torch.nn.Sequential(
            nn.Linear(dim_output, dim_ff),
            torch.nn.ReLU(),
            nn.Linear(dim_ff, dim_output)
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.norm1 = LayerNorm(dim_output)
        self.norm2 = LayerNorm(dim_output)
        self.residual1 = (dim_input == dim_output)

    def forward(self, x, mask):
        # self-attention sublayer
        if self.residual1:
            y = self.norm1(x + self.dropout(self.self_attn(x, x, mask)))
        else:
            y = self.norm1(self.dropout(self.self_attn(x, x, mask)))

        # feed forward sublayer
        z = self.norm2(y + self.dropout(self.ff(y)))

        return z


class Encoder(nn.Module):
    def __init__(self, dim_input, dim_output, n_heads,
                 dropout_rate=0.1, dim_ff=512,
                 n_blocks=1, max_pos_distance=1000):
        super(Encoder, self).__init__()
        self.positional_encoding = \
            PositionalEncoding(dim_input, dropout_rate, max_pos_distance)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(dim_input, dim_output,
                             n_heads, dropout_rate, dim_ff)
            ] + [
                EncoderBlock(dim_output, dim_output,
                             n_heads, dropout_rate, dim_ff)
                for i in range(n_blocks - 1)
            ]
        )

    def forward(self, x, lens):
        x = self.positional_encoding(x)

        mask = make_mask(x, lens)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)

        return x


class Connection(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads,
                 dropout_rate=0.1, dim_ff=512,
                 max_pos_distance=1000):
        super(Connection, self).__init__()
        if dim_out % n_heads != 0:
            raise ValueError(
                'MultiHeadSelfAttention: Output dimensions {dim_out} isn\'t'
                'a multiplier of number of heads {n_heads}'.format(
                    dim_out=dim_out,
                    n_heads=n_heads
                )
            )

        self.n_heads = n_heads
        self.dim_head = dim_out // n_heads
        self.linear_q = nn.Linear(dim_in, dim_out)
        self.linear_k = nn.Linear(dim_in, dim_out)
        self.linear_v = nn.Linear(dim_in, dim_out)
        self.linear_o = nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.b_prev = torch.nn.Parameter(torch.zeros(1, 1, dim_in))
        self.b_self = torch.nn.Parameter(torch.zeros(1, 1, dim_in))
        self.weight_softmax = torch.nn.Softmax(-1)

    def forward(self, seq1, seq2, seq_len1):
        """
        Args:
            seq1 (FloatTensor): (batch_size, len1, dim_in)
            seq2 (FloatTensor): (batch_size, len2, dim_in)
            seq_len1 (int): Length of sequences in seq1.

        Returns:
            (batch_size, query_len, dim_out)
        """
        batch_size, len1, dim_feature = seq1.shape
        batch_size, len2, dim_feature = seq2.shape

        query = seq2 + self.b_self
        key = torch.cat([seq1 + self.b_prev, seq2 + self.b_self], 1)
        value = torch.cat([seq1, seq2], 1)

        q = self.linear_q(query)
        k, v = self.linear_k(key), self.linear_v(value)
        q, k, v = [
            x.reshape(batch_size, -1, self.n_heads, self.dim_head)
             .transpose(1, 2)
            for x in (q, k, v)
        ]
        # q.shape == (batch_size, n_heads, len2, dim_head)
        # k.shape == v.shape == (batch_size, n_heads, len1 + len2, dim_head)

        # calculate attention score
        score_prev = (torch.matmul(q, k[:, :, :len1].transpose(-1, -2))
                      / np.sqrt(self.dim_head))
        score_self = ((q * k[:, :, -len2:]).sum(-1, keepdim=True)
                      / np.sqrt(self.dim_head))
        # score_prev.shape == (batch_size, n_heads, len2, len1)
        # score_self.shape == (batch_size, n_heads, len2, 1)

        # mask out padding and apply attention weights
        mask = make_mask(seq1, seq_len1)
        score_prev.masked_fill_(
            mask.view(batch_size, 1, 1, len1) == 0, -math.inf
        )
        score = torch.cat([score_prev, score_self], -1)
        weights = self.weight_softmax(score)
        weights = self.dropout(weights)
        # weight.shape == (batch_size, n_heads, len2, len1 + 1)

        attention_prev = torch.matmul(weights[:, :, :, :-1], v[:, :, :len1])
        # attention_prev.shape == (batch_size, n_heads, len2, dim_head)
        attention_self = weights[:, :, :, -1:] * v[:, :, -len2:, :]
        # attention_prev.shape == attention_self.shape == (batch_size, n_heads, len2, dim_head)

        attention = attention_prev + attention_self

        # reshape to concat results
        attention = attention.transpose(1, 2) \
                             .reshape(batch_size, len2,
                                      self.n_heads * self.dim_head)
        output = self.linear_o(attention)
        return output


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe.require_grad = False

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # return x


class Seq2Vec(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_in, dim_out, n_heads, dropout_rate,
                 pooling='attention'):
        super(Seq2Vec, self).__init__()
        if pooling not in ['attention', 'mean', 'max']:
            raise Exception('pooling method {} is not defined.'
                            .format(pooling))

        if pooling in ['mean', 'max'] and dim_in != dim_out:
            raise Exception('dim_in must equal to dim_out for {} pooling'
                            .format(pooling))

        self.pooling = pooling
        if pooling == 'attention':
            self.attention = MultiHeadAttention(dim_in, dim_out, n_heads,
                                                dropout_rate)

            q = torch.zeros(1, 1, dim_out)
            torch.nn.init.normal_(q)
            self.q = torch.nn.Parameter(q)

    def forward(self, seq, seq_len):
        """
        Args:
            seq (FloatTensor)
            seq_len (list of int)
        Returns:
            FloatTensor of shape (batch, dim_out).
        """
        if self.pooling == 'attention':
            mask = make_mask(seq, seq_len)
            q = torch.cat([self.q] * seq.shape[0], 0)
            vec = self.attention(q, seq, mask).squeeze(1)
        elif self.pooling == 'max':
            vec = seq.max(-2)[0]
        elif self.pooling == 'mean':
            vec = (
                seq.sum(-2)
                / torch.tensor(seq_len).float().to(seq.device)
                                       .unsqueeze(-1)
            )

        return vec


def make_mask(seq, lens):
    mask = torch.zeros_like(seq[:, :, 0])
    for i, ll in enumerate(lens):
        mask[i, :ll] = 1

    return mask
