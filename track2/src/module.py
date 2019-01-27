import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import SEED, DEVICE
from beam_search import Searcher


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
VALID_ACTIVATION = [a for a in dir(nn.modules.activation)
                    if not a.startswith('__')
                    and a not in ['torch', 'warnings', 'F', 'Parameter', 'Module']]


class Activation(nn.Module):
    def __init__(self, activation, *args, **kwargs):
        super(Activation, self).__init__()

        if activation in VALID_ACTIVATION:
            self.activation = \
                getattr(nn.modules.activation, activation)(*args, **kwargs)
        else:
            raise ValueError(
                f'Activation: {activation} is not a valid activation function')

    def forward(self, x):
        return self.activation(x)


class Embedding(nn.Module):
    def __init__(self, vocab, dropout):
        super(Embedding, self).__init__()

        self.emb_dim = vocab.word.emb_dim
        self.vocab_size = len(vocab.word)

        try:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.word.emb))
        except AttributeError:
            self.embedding = nn.Embedding(
                self.vocab_size, self.emb_dim, padding_idx=vocab.word.sp.pad.idx)
            print('[-] Random initialize embedding')
        else:
            print('[-] Using pretrained embedding')
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):
        emb = self.embedding(x)
        if self.dropout:
            emb = self.dropout(emb)

        return emb

    # def weighted_sum(self, weight):
    #     return (weight.unsqueeze(-1) * self.embedding.weight).sum(-2)


class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, bidirectional, dropout,
                 gumbel_embedding=False):
        super(Encoder, self).__init__()

        self.embedding = embedding
        self.gru = nn.GRU(
            self.embedding.emb_dim, hidden_size, num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0), bidirectional=bidirectional)
        self.gumbel_embedding = gumbel_embedding

    def forward(self, x):
        if not self.gumbel_embedding:
            emb = self.embedding(x)
        else:
            # TODO: Use weighted sum
            orig_shape = x.shape
            x = x.reshape(-1, self.embedding.vocab_size)
            x = F.gumbel_softmax(x, hard=True).reshape(*orig_shape)
            m, i = x.max(dim=-1)
            emb = m.unsqueeze(-1) * self.embedding(i)
        output, hidden = self.gru(emb)

        return output, hidden


class ConvEncoder(nn.Module):
    def __init__(self, embedding, kernel_sizes, filters):
        super(ConvEncoder, self).__init__()
        self.embedding = embedding
        self.kernel_sizes = kernel_sizes
        self.pads = nn.ModuleList([
            nn.ConstantPad1d(((kz - 1) // 2, kz // 2), 0) for kz in kernel_sizes])
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding.emb_dim,
                out_channels=nf,
                kernel_size=kz)
            for kz, nf in zip(kernel_sizes, filters)])

    def forward(self, x):
        emb = self.embedding(x).permute(1, 2, 0)
        o = torch.cat(
            [c(p(emb)) for p, c in zip(self.pads, self.convs)], dim=1).permute(2, 0, 1)
        h = o.mean(dim=0)

        return o, h


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, memory_size, output_size):
        super(AdditiveAttention, self).__init__()

        self.hidden_linear = nn.Linear(hidden_size, output_size)
        self.memory_linear = nn.Linear(memory_size, output_size)
        self.output_linear = nn.Linear(output_size, 1)

    def forward(self, hidden, memory, memory_pad_mask):
        batch_size = hidden.shape[1]

        hidden = hidden.transpose(0, 1).reshape(batch_size, -1)
        x = self.hidden_linear(hidden) + self.memory_linear(memory)
        score = self.output_linear(x.tanh())
        # score *= memory_pad_mask.unsqueeze(-1).float()
        score.masked_fill_(memory_pad_mask.unsqueeze(-1) == 0, float('-inf'))
        attention = F.softmax(score, dim=0)

        return attention


class Coattention(nn.Module):
    def __init__(self, C_input_size, Q_input_size):
        super(Coattention, self).__init__()

        self.Q_linear = nn.Linear(Q_input_size, C_input_size)
        self.linear = nn.Linear(C_input_size * 3, 1)

    def forward(self, C, Q, C_mask, Q_mask):
        C = C.transpose(0, 1)
        Q = Q.transpose(0, 1)
        C_mask = C_mask.transpose(0, 1)
        Q_mask = Q_mask.transpose(0, 1)

        c_len, q_len = C.shape[1], Q.shape[1]

        Q = self.Q_linear(Q)
        C_expanded = C.unsqueeze(2).expand(-1, -1, q_len, -1)
        Q_expanded = Q.unsqueeze(1).expand(-1, c_len, -1, -1)

        x = torch.cat((C_expanded, Q_expanded, C_expanded * Q_expanded), dim=-1)
        S = self.linear(x).squeeze(-1)
        S_row = S.masked_fill(Q_mask.unsqueeze(1) == 0, -np.inf)
        S_row = F.softmax(S_row, dim=2)
        S_col = S.masked_fill(C_mask.unsqueeze(2) == 0, -np.inf)
        S_col = F.softmax(S_col, dim=1)

        A = S_row @ Q
        B = S_row @ S_col.transpose(1, 2) @ C
        CQ = torch.cat((C, A, C * A, C * B), dim=-1)

        CQ = CQ.transpose(0, 1)

        return CQ


class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, c_memory_size, f_memory_size,
                 d_model, latent_size, attn_type, dropout, max_de_seq_len, sos_idx,
                 eos_idx):
        super(Decoder, self).__init__()

        self.attn_type = attn_type

        self.embedding = embedding
        if self.attn_type == 0:
            self.gru = nn.GRU(
                embedding.emb_dim + latent_size,
                hidden_size, num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0))
        if self.attn_type in [1, 4]:
            self.c_attention = AdditiveAttention(
                num_layers * hidden_size, c_memory_size, d_model)
            self.gru = nn.GRU(
                embedding.emb_dim + c_memory_size + latent_size,
                hidden_size, num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0))
        elif self.attn_type in [2, 3]:
            self.c_attention = AdditiveAttention(
                num_layers * hidden_size, c_memory_size, d_model)
            self.f_attention = AdditiveAttention(
                num_layers * hidden_size, f_memory_size, d_model)
            self.gru = nn.GRU(
                embedding.emb_dim + c_memory_size + f_memory_size + latent_size,
                hidden_size, num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0))
        self.activation = Activation('ReLU')
        self.projection_linear = nn.Linear(hidden_size, embedding.vocab_size)

        self.max_de_seq_len = max_de_seq_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, c_memory, c_pad_mask, f_memory, f_pad_mask, hidden, cf_attn,
                latent_code, target=None, scheduled_sampling_ratio=0):
        if latent_code is not None:
            latent_code = latent_code.unsqueeze(0)

        batch_size = hidden.shape[1]
        pre_output = None
        logits = []
        for step in range(self.max_de_seq_len):
            if step == 0:
                x = torch.full(
                    (1, batch_size), self.sos_idx, dtype=torch.long, device=DEVICE)
            elif target is not None and random.random() > scheduled_sampling_ratio:
                x = target[step - 1].unsqueeze(0)
            else:
                x = pre_output

            emb = self.embedding(x)
            if self.attn_type != 0:
                c_attn = self.c_attention(hidden, c_memory, c_pad_mask)
                c_context = (c_attn * c_memory).sum(dim=0, keepdim=True)
                if self.attn_type in [2, 3]:
                    f_attn = self.f_attention(hidden, f_memory, f_pad_mask)
                    if self.attn_type == 3:
                        f_attn = (f_attn + cf_attn) / 2
                    f_context = (f_attn * f_memory).sum(dim=0, keepdim=True)
                if latent_code is not None:
                    if self.attn_type in [1, 4]:
                        x = torch.cat((emb, c_context, latent_code), dim=-1)
                    elif self.attn_type in [2, 3]:
                        x = torch.cat((emb, c_context, f_context, latent_code), dim=-1)
                else:
                    if self.attn_type in [1, 4]:
                        x = torch.cat((emb, c_context), dim=-1)
                    elif self.attn_type in [2, 3]:
                        x = torch.cat((emb, c_context, f_context), dim=-1)
            else:
                x = emb
            output, hidden = self.gru(x, hidden.contiguous())
            logit = self.projection_linear(self.activation(output))
            logits.append(logit)
            _, pre_output = logit.max(dim=-1)
        logits = torch.cat(logits).transpose(0, 1)

        return logits

    def infer(self, c_memory, c_pad_mask, f_memory, f_pad_mask, hidden, cf_attn,
              latent_code, beam_size):
        if latent_code is not None:
            latent_code = latent_code.unsqueeze(0)

        batch_size = hidden.shape[1]
        predictions = []
        for bidx in range(batch_size):
            searcher = Searcher(beam_size, self.max_de_seq_len, self.eos_idx)
            searcher.init()
            x = torch.full((1, 1), self.sos_idx, dtype=torch.long, device=DEVICE)
            h = hidden[:, bidx, :].unsqueeze(1)
            while not searcher.end:
                emb = self.embedding(x)
                if self.attn_type != 0:
                    c_attn = self.c_attention(
                        h, c_memory[:, bidx, :].unsqueeze(1),
                        c_pad_mask[:, bidx].unsqueeze(-1))
                    c_context = (c_attn * c_memory[:, bidx, :].unsqueeze(1)).sum(dim=0, keepdim=True)
                    if self.attn_type in [2, 3]:
                        f_attn = self.f_attention(
                            h, f_memory[:, bidx, :].unsqueeze(1),
                            f_pad_mask[:, bidx].unsqueeze(-1))
                        if self.attn_type == 3:
                            f_attn = (f_attn + cf_attn[:, bidx, :].unsqueeze(1)) / 2
                        f_context = (f_attn * f_memory[:, bidx, :].unsqueeze(1)).sum(dim=0, keepdim=True)
                    if latent_code is not None:
                        lc = latent_code[:, bidx, :].unsqueeze(1).repeat(1, emb.shape[1], 1)
                        if self.attn_type in [1, 4]:
                            x = torch.cat((emb, c_context, lc), dim=-1)
                        elif self.attn_type in [2, 3]:
                            x = torch.cat((emb, c_context, f_context, lc), dim=-1)
                    else:
                        if self.attn_type in [1, 4]:
                            x = torch.cat((emb, c_context), dim=-1)
                        elif self.attn_type in [2, 3]:
                            x = torch.cat((emb, c_context, f_context), dim=-1)
                else:
                    x = emb
                output, h = self.gru(x, h.contiguous())
                output = output.squeeze(0)
                output = self.projection_linear(self.activation(output))
                output = F.log_softmax(output, dim=-1)
                best_logits, beam_indices = searcher.step(output)
                if best_logits is None:
                    break
                x = best_logits.t()
                h = h[:, beam_indices, :]
            predictions.append(searcher.sequences[0][0])

        return predictions


class Net(nn.Module):
    def __init__(self, vocab, c_en_hidden_size, c_en_num_layers, c_en_bidirectional,
                 f_en_type, f_en_hidden_size, f_en_num_layers, f_en_bidirectional,
                 f_en_kernel_sizes, f_en_filters, r_en_hidden_size, r_en_num_layers,
                 r_en_bidirectional, de_hidden_size, de_num_layers, d_model, dropout,
                 max_de_seq_len, latent_size, attn_type):
        super(Net, self).__init__()

        self.attn_type = attn_type
        c_total_hidden_size = \
            c_en_num_layers * (c_en_bidirectional + 1) * c_en_hidden_size
        if f_en_type == "RNN":
            f_total_output_size = \
                (f_en_bidirectional + 1) * f_en_hidden_size
            f_total_hidden_size = \
                f_en_num_layers * (f_en_bidirectional + 1) * f_en_hidden_size
        else:
            f_total_output_size = f_total_hidden_size = sum(f_en_filters)
        r_total_hidden_size = \
            r_en_num_layers * (r_en_bidirectional + 1) * r_en_hidden_size

        embedding = Embedding(vocab, dropout)

        self.c_encoder = Encoder(
            embedding, c_en_hidden_size, c_en_num_layers, c_en_bidirectional, dropout)
        if f_en_type == "RNN":
            self.f_encoder = Encoder(
                embedding, f_en_hidden_size, f_en_num_layers, f_en_bidirectional,
                dropout)
        elif f_en_type == "CNN":
            self.f_encoder = ConvEncoder(
                embedding, f_en_kernel_sizes, f_en_filters)

        self.coattention = Coattention(
            c_en_hidden_size * (c_en_bidirectional + 1), f_total_output_size)
        if self.attn_type == 4:
            self.coattention_encoder = nn.GRU(
                c_en_hidden_size * 8, c_en_hidden_size, num_layers=c_en_num_layers,
                bidirectional=c_en_bidirectional, dropout=dropout)

        self.latent_size = latent_size
        if latent_size > 0:
            self.r_encoder = Encoder(
                embedding, r_en_hidden_size, r_en_num_layers, r_en_bidirectional,
                dropout)
            if self.attn_type in [0, 1, 4]:
                input_size = c_total_hidden_size + r_total_hidden_size
            elif self.attn_type in [2, 3]:
                input_size = \
                    c_total_hidden_size + f_total_hidden_size + r_total_hidden_size
            self.latent_mean_linear = nn.Linear(input_size, latent_size)
            self.latent_log_var_linear = nn.Linear(input_size, latent_size)

        self.f_linear = nn.Linear(f_total_output_size, d_model)
        self.c_linear = nn.Linear(
            (c_en_bidirectional + 1) * c_en_hidden_size, d_model)
        self.cf_attn_linear = nn.Linear(d_model, 1)

        self.de_hidden_size = de_hidden_size
        self.de_num_layers = de_num_layers
        self.hidden_projector = nn.Linear(
            c_total_hidden_size, de_num_layers * de_hidden_size)

        self.decoder = Decoder(
            embedding, de_hidden_size, de_num_layers,
            (c_en_bidirectional + 1) * c_en_hidden_size,
            f_total_output_size, d_model, latent_size, self.attn_type,
            dropout, max_de_seq_len, vocab.word.sp.sos.idx, vocab.word.sp.eos.idx)

    def forward(self, c, f, r, c_pad_mask, f_pad_mask, r_pad_mask,
                scheduled_sampling_ratio=0):
        batch_size = c.shape[0]
        c = c.transpose(0, 1)
        f = f.transpose(0, 1)
        r = r.transpose(0, 1)
        c_pad_mask = c_pad_mask.transpose(0, 1)
        f_pad_mask = f_pad_mask.transpose(0, 1)
        r_pad_mask = r_pad_mask.transpose(0, 1)

        c_en_output, c_hidden = self.c_encoder(c)
        f_en_output, f_hidden = self.f_encoder(f)
        if self.attn_type == 4:
            c_en_output = self.coattention(
                c_en_output, f_en_output, c_pad_mask, f_pad_mask)
            c_en_output, c_hidden = self.coattention_encoder(c_en_output)

        if self.latent_size > 0:
            _, r_hidden = self.r_encoder(r)
            c_hidden_flat = c_hidden.transpose(0, 1).reshape(batch_size, -1)
            f_hidden_flat = f_hidden.transpose(0, 1).reshape(batch_size, -1)
            r_hidden_flat = r_hidden.transpose(0, 1).reshape(batch_size, -1)
            if self.attn_type in [0, 1, 4]:
                latent_input = \
                    torch.cat((c_hidden_flat, r_hidden_flat), dim=-1)
            elif self.attn_type in [2, 3]:
                latent_input = \
                    torch.cat((c_hidden_flat, f_hidden_flat, r_hidden_flat), dim=-1)
            latent_mean = self.latent_mean_linear(latent_input)
            latent_log_var = self.latent_log_var_linear(latent_input)
            std = torch.exp(0.5 * latent_log_var)
            eps = torch.randn([batch_size, self.latent_size], device=DEVICE)
            latent_code = eps * std + latent_mean
        else:
            latent_mean = latent_log_var = latent_code = None

        de_init_hidden = (
            self.hidden_projector(c_hidden.transpose(0, 1).reshape(batch_size, -1))
            .reshape(batch_size, self.de_num_layers, self.de_hidden_size)
            .transpose(0, 1))

        f_en_output_ = self.f_linear(f_en_output).transpose(0, 1).unsqueeze(2)
        c_en_output_ = self.c_linear(c_en_output).transpose(0, 1).unsqueeze(1)
        score = self.cf_attn_linear((f_en_output_ + c_en_output_).tanh()).squeeze()
        score.masked_fill_(f_pad_mask.t().unsqueeze(-1) == 0, float('-inf'))
        cf_attn = F.softmax(score.sum(dim=-1), dim=-1).t().unsqueeze(-1)

        logits = self.decoder(
            c_en_output, c_pad_mask, f_en_output, f_pad_mask, de_init_hidden, cf_attn,
            latent_code, target=r, scheduled_sampling_ratio=scheduled_sampling_ratio)

        return logits, latent_mean, latent_log_var

    def infer(self, c, f, c_pad_mask, f_pad_mask, beam_size=1):
        batch_size = c.shape[0]
        c = c.transpose(0, 1)
        f = f.transpose(0, 1)
        c_pad_mask = c_pad_mask.transpose(0, 1)
        f_pad_mask = f_pad_mask.transpose(0, 1)

        c_en_output, c_hidden = self.c_encoder(c)
        f_en_output, f_hidden = self.f_encoder(f)

        # attn-type: 3
        f_en_output_ = self.f_linear(f_en_output).transpose(0, 1).unsqueeze(2)
        c_en_output_ = self.c_linear(c_en_output).transpose(0, 1).unsqueeze(1)
        score = self.cf_attn_linear((f_en_output_ + c_en_output_).tanh()).squeeze()
        score.masked_fill_(f_pad_mask.t().unsqueeze(-1) == 0, float('-inf'))
        cf_attn = F.softmax(score.sum(dim=-1), dim=-1).t().unsqueeze(-1)

        # attn-type: 4
        if self.attn_type == 4:
            c_en_output = self.coattention(
                c_en_output, f_en_output, c_pad_mask, f_pad_mask)
            c_en_output, c_hidden = self.coattention_encoder(c_en_output)

        de_init_hidden = (
            self.hidden_projector(c_hidden.transpose(0, 1).reshape(batch_size, -1))
            .reshape(batch_size, self.de_num_layers, self.de_hidden_size)
            .transpose(0, 1))

        if self.latent_size > 0:
            latent_code = torch.randn([batch_size, self.latent_size], device=DEVICE)
        else:
            latent_code = None

        predictions = self.decoder.infer(
            c_en_output, c_pad_mask, f_en_output, f_pad_mask, de_init_hidden, cf_attn,
            latent_code, beam_size)

        return predictions, None, None
