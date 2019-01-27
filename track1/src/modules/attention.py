import torch
import math

from torch import nn
from torch.nn import functional as F


class CoAttention(nn.Module):
    def __init__(self, dim_embeddings, pooling='mean', use_projection=True):
        super(CoAttention, self).__init__()
        
        self.dim_e = dim_embeddings
        self.use_projection = use_projection

        if pooling.lower() not in ['mean', 'max', 'align']:
            raise ValueError("Pooling type {} not supported".format(pooling))
        self.pooling = pooling.lower()

        if use_projection:
            self.M = nn.Linear(self.dim_e, self.dim_e)
        self.fcc = nn.Sequential(nn.Linear(2 * dim_embeddings, 1), nn.ReLU())
        self.fcm = nn.Sequential(nn.Linear(dim_embeddings, 1), nn.ReLU())
        self.fcs = nn.Sequential(nn.Linear(dim_embeddings, 1), nn.ReLU())
        
    def forward(self, query, query_lens, option, option_lens):
        # print('query.size', query.size())
        # print('option.size', option.size())
        # query.size() = (batch_size, query_len, dim_e)
        # option.size() = (batch_size, option_len, dim_e)
        if self.use_projection:
            affinity = torch.matmul(self.M(query), option.transpose(1, 2))
        else:
            affinity = torch.matmul(query, option.transpose(1, 2))
        # print('affinity.size', affinity.size())
        # affinity.size() = (batch_size, query_len, option_len)
        mask_query = make_mask(query, query_lens)
        # mask_query.size() = (batch_size, query_len)
        mask_option = make_mask(option, option_lens)
        # mask_option.size() = (batch_size, option_len)

        masked_affinity_query = affinity.masked_fill(
            mask_query.unsqueeze(2) == 0, -math.inf)
        masked_affinity_option = affinity.masked_fill(
            mask_option.unsqueeze(1) == 0, -math.inf)

        if self.pooling == 'mean':
            query_weights = F.softmax(masked_affinity_query.mean(2), dim=-1)
            summary_query = query_weights.unsqueeze(2) * query
            option_weights = F.softmax(masked_affinity_option.mean(1), dim=-1)
            summary_option = option_weights.unsqueeze(2) * option
        elif self.pooling == 'max':
            query_weights = F.softmax(
                masked_affinity_query.max(2)[0], dim=-1)
            summary_query = query_weights.unsqueeze(2) * query
            option_weights = F.softmax(
                masked_affinity_option.max(1)[0], dim=-1)
            summary_option = option_weights.unsqueeze(2) * option
        else:
            query_weights = F.softmax(masked_affinity_option, dim=2)
            summary_query = torch.matmul(query_weights, option)
            option_weights = F.softmax(masked_affinity_query, dim=1)
            summary_option = torch.matmul(option_weights.transpose(1, 2),
                                          query)

        # print('summary_query.size', summary_query.size())
        # print('summary_option.size', summary_option.size())
        query_c = self.fcc(torch.cat([summary_query, query], -1))
        query_m = self.fcm(summary_query * query)
        query_s = self.fcs(summary_query - query)
        option_c = self.fcc(torch.cat([summary_option, option], -1))
        option_m = self.fcm(summary_option * option)
        option_s = self.fcs(summary_option - option)

        return (torch.cat([query_c, query_m, query_s], -1),
                torch.cat([option_c, option_m, option_s], -1))


class CoAttentionEncoder(nn.Module):
    def __init__(self, dim_embeddings, use_projection=True):
        super(CoAttentionEncoder, self).__init__()

        self.co_att_max = CoAttention(
            dim_embeddings, pooling='max', use_projection=use_projection)
        self.co_att_mean = CoAttention(
            dim_embeddings, pooling='mean', use_projection=use_projection)
        self.co_att_align = CoAttention(
            dim_embeddings, pooling='align', use_projection=use_projection)

    def forward(self, query, query_lens, option, option_lens):
        att_q_max, att_o_max = self.co_att_max(
            query, query_lens, option, option_lens)
        att_q_mean, att_o_mean = self.co_att_mean(
            query, query_lens, option, option_lens)
        att_q_align, att_o_align = self.co_att_align(
            query, query_lens, option, option_lens)

        new_query = torch.cat([query, att_q_max, att_q_mean, att_q_align], -1)
        new_option = torch.cat([option, att_o_max, att_o_mean, att_o_align], -1)

        return new_query, new_option


class IntraAttention(nn.Module):
    def __init__(self, dim_embeddings, use_projection=True):
        super(IntraAttention, self).__init__()
        
        self.dim_e = dim_embeddings
        self.use_projection = use_projection
        
        if use_projection:
            self.M = nn.Linear(self.dim_e, self.dim_e)
        self.fcc = nn.Sequential(nn.Linear(2 * dim_embeddings, 1), nn.ReLU())
        self.fcm = nn.Sequential(nn.Linear(dim_embeddings, 1), nn.ReLU())
        self.fcs = nn.Sequential(nn.Linear(dim_embeddings, 1), nn.ReLU())
        
    def forward(self, query, query_lens):
        # print('query.size', query.size())
        # query.size() = (batch_size, query_len, dim_e)
        if self.use_projection:
            affinity = torch.matmul(self.M(query), query.transpose(1, 2))
        else:
            affinity = torch.matmul(query, query.transpose(1, 2))
        # print('affinity.size', affinity.size())
        # affinity.size() = (batch_size, query_len, query_len)
        mask_query = make_mask(query, query_lens)
        # mask_query.size() = (batch_size, query_len)

        masked_affinity_query = affinity.masked_fill(
            mask_query.unsqueeze(2) == 0, -math.inf)

        query_weights = F.softmax(masked_affinity_query, dim=1)
        summary_query = torch.matmul(query_weights.transpose(1, 2),
                                     query)

        # print('summary_query.size', summary_query.size())
        query_c = self.fcc(torch.cat([summary_query, query], -1))
        query_m = self.fcm(summary_query * query)
        query_s = self.fcs(summary_query - query)

        return torch.cat([query_c, query_m, query_s], -1)


def make_mask(seq, lens):
    mask = torch.zeros_like(seq[:, :, 0])
    for i, ll in enumerate(lens):
        mask[i, :ll] = 1

    return mask
