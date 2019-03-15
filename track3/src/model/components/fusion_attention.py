import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import copy

class FullyAttention(nn.Module):
    '''
    Attention proposed in FusionNet
    '''
    def __init__(self, in_dim, attn_dim):
        super(FullyAttention, self).__init__()
        self.U = nn.Linear(in_dim, attn_dim)

        self.D = nn.Parameter(torch.Tensor(attn_dim))
        stdv = 1. / math.sqrt(self.D.size(0))
        self.D.data.uniform_(-stdv, stdv)

    def forward(self, h_output, h_context, context):
        '''
        Fusing output to context
        Args:
            h_output: (batch, out_len, in_dim)
            h_context: (batch, con_len, in_dim)
            context: (batch, con_len, hidden_dim)
        Return:
            mix: (batch, out_len, hidden_dim)
        '''
        batch = h_output.size(0)
        in_dim = h_output.size(2)
        con_len = h_context.size(1)

        # (batch, len, in_dim) -> (batch, len, attn_dim)
        o_feature = F.relu(self.U(h_output))
        c_feature = F.relu(self.U(h_context))
        # (batch, out_len, attn_dim) * (attn_dim) -> (batch, out_len, attn_dim)
        o_feature *= self.D
        # (batch, out_len, attn_dim) * (batch, con_len, attn_dim) -> (batch, out_len, con_len)
        attn = torch.bmm(o_feature, c_feature.transpose(1, 2))
        attn = F.softmax(attn.view(-1, con_len), -1).view(batch, -1, con_len)
        # (batch, out_len, con_len) * (batch, con_len, hidden_dim) -> (batch, out_len, hidden_dim)
        mix = torch.bmm(attn, context)
        return mix

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        '''
        Args:
            query: (batch, q_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
        '''
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class WordAttention(nn.Module):
    '''
    Word-level attention
    '''
    def __init__(self, dim):
        '''
        Args:
            dim: input_dim = hidden_size = output_dim
        '''
        super(WordAttention, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, c, q):
        '''
        Args:
            c: (batch, c_len, dim)
            q: (batch, q_len, dim)
        Output:
            mix: (batch, c_len, dim)
        '''
        batch = c.size(0)
        word_dim = c.size(2)
        q_len = q.size(1)

        c_feature = F.relu(self.linear(c))
        q_feature = F.relu(self.linear(q))
        # (batch, c_len, dim) * (batch, q_len, dim) -> (batch, c_len, q_len)
        attn = torch.bmm(c_feature, q_feature.transpose(1, 2))
        attn = F.softmax(attn.view(-1, q_len), -1).view(batch, -1, q_len)
        # (batch, c_len, q_len) * (batch, q_len, dim) -> (batch, c_len, dim)
        mix = torch.bmm(attn, q)
        return mix

if __name__ == '__main__':
    attn = MultiHeadedAttention(8, 64)
    q = torch.Tensor(32, 4, 64)
    v = torch.Tensor(32, 100, 64)
    print(attn(q, v, v).size())
