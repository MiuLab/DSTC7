import math
import torch
import pdb
# from .transformer import EncoderLayer as TransformerEncoder
from .transformer2 import Encoder as TransformerEncoder
from .attention import CoAttentionEncoder, IntraAttention


class DualRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product'):
        super(DualRNN, self).__init__()
        self.context_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]

    def forward(self, context, context_lens, options, option_lens):
        context_hidden = self.context_encoder(context, context_lens)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class HierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1):
        super(HierRNN, self).__init__()
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        context_hidden = self.context_encoder(context, context_ends)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class UttHierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1,
                 dropout_rate=0.0, utt_enc_type='rnn'):
        super(UttHierRNN, self).__init__()
        """
        self.utterance_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden,
                                              self.utterance_encoder.rnn)
        """
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden,
                                              utt_enc_type=utt_enc_type)
        self.utterance_encoder = self.context_encoder.utt_enc
        self.dropout_ctx_encoder = torch.nn.Dropout(p=dropout_rate)
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct(),
            'mlp': BatchMLP(dim_hidden * 2)
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        context_last = [
            list(context[i, (ends[-2] if len(ends)>1 else 0):ends[-1]])
            for i, ends in enumerate(context_ends)
        ]
        padding = torch.zeros_like(context[0, 0]).cuda()
        context_lens, context = pad_and_cat(context_last, padding)
        context_hidden = self.utterance_encoder(context)
        context_hidden = self.dropout_ctx_encoder(context_hidden)
        predict_option = self.transform(context_hidden)
        """
        context_hidden = self.context_encoder(context, context_ends)
        context_hidden = self.dropout_ctx_encoder(context_hidden)
        predict_option = self.transform(context_hidden)
        """
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.utterance_encoder(option)
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class UttBinHierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1,
                 dropout_rate=0.0, utt_enc_type='rnn',
                 use_co_att=False, use_intra_att=False,
                 only_last_context=False):
        super(UttBinHierRNN, self).__init__()
        """
        self.utterance_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden,
                                              self.utterance_encoder.rnn)
        """
        self.dim_embeddings = self.dim_features = dim_embeddings

        self.use_intra_att = use_intra_att
        if use_intra_att:
            self.intra_att_context = IntraAttention(dim_embeddings)
            self.intra_att_option = IntraAttention(dim_embeddings)
            self.dim_features += 3

        self.use_co_att = use_co_att
        if use_co_att:
            self.co_att_encoder = CoAttentionEncoder(dim_embeddings)
            self.dim_features += 9

        self.only_last_context = only_last_context
        if not only_last_context:
            self.context_encoder = HierRNNEncoder(self.dim_features,
                                                  dim_hidden, dim_hidden,
                                                  utt_enc_type=utt_enc_type)
            self.utterance_encoder = self.context_encoder.utt_enc
        else:
            if utt_enc_type == 'rnn':
                self.utterance_encoder = LSTMEncoder(self.dim_features,
                                                     dim_hidden)
            else:
                self.utterance_encoder = Conv1dEncoder(
                    dim_embeddings,
                    dim_hidden // 2,
                    kernel_sizes=[2, 3, 4, 5])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(32, 1, bias=False)
        )
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        """
        self.sims = [
            torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            BatchInnerProduct()
        ]
        """

        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        context_lens = [ends[-1] for ends in context_ends]
        
        if self.only_last_context:
            context_last = [
                list(context[i, (ends[-2] if len(ends)>1 else 0):ends[-1]])
                for i, ends in enumerate(context_ends)
            ]
            padding = torch.zeros_like(context[0, 0]).cuda()
            context_lens, context = pad_and_cat(context_last, padding)

        if self.use_intra_att:
            context_intra = self.intra_att_context(context, context_lens)
        
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_len = [ol[i] for ol in option_lens]

            if self.use_co_att:
                context_cast, option_cast = self.co_att_encoder(
                    context, context_lens, option, option_len)
            else:
                context_cast = context
                option_cast = option

            if self.use_intra_att:
                option_intra = self.intra_att_option(option, option_len)
                context_cast = torch.cat([context_cast, context_intra], -1)
                option_cast = torch.cat([option_cast, option_intra], -1)

            # context_hidden = self.context_encoder(context_cast, context_ends)
            context_hidden = self.utterance_encoder(context_cast)
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.utterance_encoder(option_cast)
            # option_concat = torch.cat([option_hidden, context_hidden, option_hidden*context_hidden], -1)
            # logit = self.mlp(option_concat)

            fused = torch.cat(
                [option_hidden*context_hidden, option_hidden-context_hidden],
                -1
            )
            """
            for sim in self.sims:
                similarity = sim(option_hidden, context_hidden).unsqueeze(-1)
                fused = torch.cat([fused, similarity], -1)
            """
            logit = self.mlp(fused)

            logit = torch.reshape(logit, (-1,))
            logits.append(logit)

        logits = torch.stack(logits, 1)
        return logits


class HierRNNEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self,
                 dim_embeddings,
                 dim_hidden1=128,
                 dim_hidden2=128,
                 utt_enc=None,
                 ctx_enc=None,
                 utt_enc_type='rnn'):
        super(HierRNNEncoder, self).__init__()

        self.utt_enc_type = utt_enc_type

        if utt_enc is not None:
            self.utt_enc = utt_enc
            if utt_enc_type == 'rnn' or utt_enc_type == 'pool-lstm':
                self.utt_enc_rnn = self.utt_enc.rnn
        else:
            if utt_enc_type == 'rnn':
                self.utt_enc = LSTMEncoder(dim_embeddings, dim_hidden1)
                self.utt_enc_rnn = self.utt_enc.rnn
            elif utt_enc_type == 'cnn':
                self.utt_enc = Conv1dEncoder(dim_embeddings,
                                             dim_hidden1 // 2,
                                             kernel_sizes=[2, 3, 4, 5])
            elif utt_enc_type == 'pool-lstm':
                self.utt_enc = LSTMPoolingEncoder(dim_embeddings, dim_hidden1)
                self.utt_enc_rnn = self.utt_enc.rnn

        if ctx_enc is not None:
            self.rnn2 = ctx_enc
        else:
            self.rnn2 = LSTMEncoder(2 * dim_hidden1, dim_hidden2)

        self.register_buffer('padding', torch.zeros(2 * dim_hidden2))

    def forward(self, seq, ends):
        batch_size = seq.size(0)

        if self.utt_enc_type == 'rnn':
            """
            end_outputs = []
            for b in range(batch_size):
                outputs = []
                for i in range(len(ends[b])-1):
                    start, end = ends[b][i], ends[b][i+1]
                    outputs.append(
                        self.utt_enc(seq[b, start:end].unsqueeze(0)).squeeze()
                    )
                end_outputs.append(outputs)
            """
            outputs, _ = self.utt_enc_rnn(seq)
            end_outputs = [[outputs[b, end] for end in ends[b]]
                           for b in range(batch_size)]
        elif self.utt_enc_type == 'cnn':
            end_outputs = []
            for b in range(batch_size):
                outputs = []
                for i in range(len(ends[b])):
                    start, end = (0 if i == 0 else ends[b][i-1]), ends[b][i]
                    outputs.append(
                        self.utt_enc(seq[b, start:end].unsqueeze(0)).squeeze()
                    )
                end_outputs.append(outputs)
        elif self.utt_enc_type == 'pool-lstm':
            end_outputs = self.utt_enc(seq, ends=ends)
        
        lens, end_outputs = pad_and_cat(end_outputs, self.padding)
        encoded = self.rnn2(end_outputs, list(map(len, ends)))
        return encoded


class BatchInnerProduct(torch.nn.Module):
    """ 

    Args:

    """

    def __init__(self):
        super(BatchInnerProduct, self).__init__()

    def forward(self, a, b):
        return (a * b).sum(-1)


class BatchMLP(torch.nn.Module):
    """

    Args:

    """
    def __init__(self, dim_in):
        super(BatchMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_in * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, a, b):
        return self.mlp(
            torch.cat([a - b, a * b], -1)
        ).squeeze(-1)



class LSTMEncoder(torch.nn.Module):
    """ 

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden):
        super(LSTMEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embeddings,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)

    def forward(self, seqs, seq_lens=None):
        _, hidden = self.rnn(seqs)
        _, c = hidden
        return c.transpose(1, 0).contiguous().view(c.size(1), -1)


class LSTMPoolingEncoder(torch.nn.Module):
    """ 

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden, pooling='meanmax'):
        super(LSTMPoolingEncoder, self).__init__()
        if pooling.lower() not in ['mean', 'max', 'meanmax']:
            raise ValueError(
                'LSTMPoolingEncoder: {} pooling not supported'.format(pooling)
            )
        self.pooling = pooling
        self.rnn = torch.nn.LSTM(dim_embeddings,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)

    def forward(self, seqs, seq_lens=None, ends=None):
        output, _ = self.rnn(seqs)
       
        if ends is None:
            if self.pooling == 'mean':
                pooled = output.mean(1)
            elif self.pooling == 'max':
                pooled = output.max(1)[0]
            else:
                pooled = torch.cat([output.mean(1), output.max(1)[0]], -1)
        else:
            pooled = []
            for b in range(seqs.size(0)):
                outputs = []
                for i in range(len(ends[b])):
                    start, end = (0 if i == 0 else ends[b][i-1]), ends[b][i]
                    # outs.size() == (seq_len, dim_hidden)
                    outs = output[b, start:end]
                    if self.pooling == 'mean':
                        outs = outs.mean(0)
                    elif self.pooling == 'max':
                        outs = outs.max(0)[0]
                    else:
                        outs = torch.cat([outs.mean(0), outs.max(0)[0]], -1)
                    outputs.append(outs)
                pooled.append(outputs)
            
        return pooled


class Conv1dEncoder(torch.nn.Module):
    def __init__(self, dim_embeddings, num_filters,
                 kernel_sizes, dropout_rate=0.0):
        super(Conv1dEncoder, self).__init__()
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(dim_embeddings,
                                num_filters,
                                kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

        self.dim_embeddings = dim_embeddings
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.max_kernel_size = max(kernel_sizes)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, seqs, seq_lens=None):
        if seqs.size(1) < self.max_kernel_size:
            pad_len = self.max_kernel_size - seqs.size(1)
            padding = torch.stack(
                [torch.zeros(self.dim_embeddings)] * pad_len, 0
            ).unsqueeze(0).repeat(seqs.size(0), 1, 1)
            seqs = torch.cat([seqs, padding.cuda()], 1)
        
        seqs = seqs.transpose(1, 2)
        encoded = []
        for conv in self.convs:
            encoded.append(
                self.activation(conv(seqs).transpose(1, 2).max(1)[0])
            )
        return self.dropout(torch.cat(encoded, 1))


class RankLoss(torch.nn.Module):
    """ 
    Args:

    """

    def __init__(self, margin=0.2, threshold=None):
        super(RankLoss, self).__init__()
        self.threshold = threshold
        self.margin = margin if threshold is not None else margin / 2
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(self.margin)

    def forward(self, logits, labels):
        positive_mask = (1 - labels).byte()
        positive_logits = logits.masked_fill(positive_mask, math.inf)
        positive_min = lse_min(positive_logits)

        negative_mask = labels.byte()
        negative_logits = logits.masked_fill(negative_mask, -math.inf)
        negative_max = lse_max(negative_logits)

        ones = torch.ones_like(negative_max)
        if self.threshold is None:
            loss = self.margin_ranking_loss(positive_min, negative_max,
                                            ones.squeeze(-1))
        else:
            loss = (self.margin_ranking_loss(positive_min,
                                             self.threshold * ones,
                                             ones.squeeze(-1))
                    + self.margin_ranking_loss(negative_max,
                                               self.threshold * ones,
                                               - ones.squeeze(-1))
                    )

        return loss.mean()


class NLLLoss(torch.nn.Module):
    """
    Args:

    """

    def __init__(self, epsilon=1e-6):
        super(NLLLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        batch_size = logits.size(0)
        probs = torch.nn.functional.softmax(logits, -1)
        loss = (-torch.log(probs + self.epsilon) *
                labels.float()).sum() / batch_size
        return loss


def pad_and_cat(tensors, padding, pad_size=-1):
    """ Pad lists to have same number of elements, and concatenate
    those elements to a 3d tensor.

    Args:
        tensors (list of list of Tensors): Each list contains
            list of operand embeddings. Each operand embedding is of
            size (dim_element,).
        padding (Tensor):
            Element used to pad lists, with size (dim_element,).

    Return:
        n_tensors (list of int): Length of lists in tensors.
        tensors (Tensor): Concatenated tensor after padding the list.
    """
    n_tensors = [len(ts) for ts in tensors]
    pad_size = max(n_tensors) if pad_size < 0 else pad_size

    # pad to has same number of operands for each problem
    tensors = [ts + (pad_size - len(ts)) * [padding]
               for ts in tensors]

    # tensors.size() = (batch_size, pad_size, dim_hidden)
    tensors = torch.stack([torch.stack(t)
                           for t in tensors], dim=0)

    return n_tensors, tensors


def pad_seqs(tensors, pad_element):
    lengths = list(map(len, tensors))
    max_length = max(lengths)
    padded = []
    for tensor, length in zip(tensors, lengths):
        if max_length > length:
            padding = torch.stack(
                [pad_element] * (max_length - length), 0
            )
            padded.append(torch.cat([tensor, padding], 0))
        else:
            padded.append(tensor)

    return lengths, torch.stack(padded, 0)


def lse_max(a, dim=-1):
    max_a = torch.max(a, dim=dim, keepdim=True)[0]
    lse = torch.log(
        torch.sum(torch.exp(a - max_a),
                  dim=dim, keepdim=True)
    ) + max_a
    return lse


def lse_min(a, dim=-1):
    return -lse_max(-a, dim)
