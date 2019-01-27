import torch
from .transformer2 import Encoder as TransformerEncoder
from .transformer2 import Connection, Seq2Vec
from .common import (pad_seqs, BatchInnerProduct, pad_and_cat)
from .attention import CoAttention, IntraAttention
from .mcan import HighwayNetwork


class RecurrentTransformer(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, n_heads, dropout_rate, dim_ff,
                 dim_encoder=102, dim_encoder_ff=256,
                 has_emb=False, vol_size=-1, n_blocks=1,
                 use_mcan=False, seq2vec_pooling='attention',
                 bi_attention='last'):
        super(RecurrentTransformer, self).__init__()
        self.bi_attention = bi_attention
        self.transformer = RecurrentTransformerEncoder(
            dim_embeddings + (12 if use_mcan else 0),
            n_heads, dropout_rate, dim_ff,
            dim_encoder, dim_encoder_ff, n_blocks)
        self.last_encoder = TransformerEncoder(
            dim_encoder, dim_encoder, n_heads,
            dropout_rate, dim_encoder_ff
        )
        self.attn = Connection(dim_encoder, dim_encoder, n_heads,
                               dropout_rate, dim_encoder_ff)
        self.seq2vec = Seq2Vec(dim_encoder, dim_encoder, n_heads, dropout_rate,
                               pooling=seq2vec_pooling)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_encoder, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.similarity = BatchInnerProduct()
        self.register_buffer('padding', torch.zeros(dim_embeddings))
        self.register_buffer('padding2', torch.zeros(dim_encoder))
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

        self.use_mcan = use_mcan
        if use_mcan:
            self.mcan = MCAN(dim_embeddings, dropout_rate)

    def forward(self, context, context_ends, options, option_lens):
        batch_size = context.size(0)

        if not self.use_mcan:
            context_enc, outputs = self.transformer(context, context_ends)
            if self.bi_attention == 'all':
                context_lens, context_enc = pad_seqs(outputs, self.padding2)
            else:
                context_lens, context_enc = pad_seqs(context_enc, self.padding2)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            opt_lens = [option_lens[b][i]
                        for b in range(batch_size)]

            if self.use_mcan:
                ctx_features, opt_features = self.mcan(context, context_ends,
                                                       option, opt_lens)
                context_cat = torch.cat([context, ctx_features], -1)
                context_enc, outputs = self.transformer(context_cat,
                                                        context_ends)
                if self.bi_attention == 'all':
                    context_lens, context_enc = pad_seqs(outputs,
                                                         self.padding2)
                else:
                    context_lens, context_enc = pad_seqs(context_enc,
                                                         self.padding2)
                option = torch.cat([option, opt_features], -1)

            option_enc = self.transformer.encoder(option[:, :max(opt_lens)],
                                                  opt_lens)
            attn_co = self.attn(context_enc, option_enc, context_lens)
            attn_oc = self.attn(option_enc, context_enc, opt_lens)

            ctx_vecs = self.seq2vec(attn_oc, opt_lens)
            opt_vecs = self.seq2vec(attn_co, context_lens)

            # logit = self.mlp(vecs).squeeze(-1)
            logit = self.similarity(ctx_vecs, opt_vecs)
            logits.append(logit)

        # pdb.set_trace()
        logits = torch.stack(logits, 1)
        return logits


class NaiveTransformer(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, n_heads, dropout_rate, dim_ff,
                 dim_encoder=102, dim_encoder_ff=256,
                 has_emb=False, vol_size=-1, n_blocks=1,
                 use_mcan=False, seq2vec_pooling='attention',
                 last_only=False):
        super(NaiveTransformer, self).__init__()
        self.last_only = last_only
        self.transformer = TransformerEncoder(
            dim_embeddings + (12 if use_mcan else 0),
            dim_encoder, n_heads, dropout_rate, dim_ff,
            n_blocks)
        self.attn = Connection(dim_encoder, dim_encoder, n_heads,
                               dropout_rate, dim_encoder_ff)
        self.seq2vec = Seq2Vec(dim_encoder, dim_encoder, n_heads, dropout_rate,
                               pooling=seq2vec_pooling)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_encoder, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.similarity = BatchInnerProduct()
        self.register_buffer('padding', torch.zeros(dim_embeddings))
        self.register_buffer('padding2', torch.zeros(dim_encoder))
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

        self.use_mcan = use_mcan
        if use_mcan:
            self.mcan = MCAN(dim_embeddings, dropout_rate)

    def forward(self, context, context_ends, options, option_lens):
        batch_size = context.size(0)
        if self.last_only:
            context = [c[(ends[-2] if len(ends) > 1 else 0):ends[-1] + 1]
                       for c, ends in zip(context, context_ends)]
            context_lens, context = pad_seqs(context, self.padding)

        context_lens = [e[-1] for e in context_ends]

        if not self.use_mcan:
            context_enc = self.transformer(context, context_lens)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            opt_lens = [option_lens[b][i]
                        for b in range(batch_size)]

            if self.use_mcan:
                ctx_features, opt_features = self.mcan(context, context_ends,
                                                       option, opt_lens)
                context_cat = torch.cat([context, ctx_features], -1)
                context_enc = self.transformer(context_cat, context_ends)
                option = torch.cat([option, opt_features], -1)

            option_enc = self.transformer(option[:, :max(opt_lens)],
                                          opt_lens)
            attn_co = self.attn(context_enc, option_enc, context_lens)
            attn_oc = self.attn(option_enc, context_enc, opt_lens)

            ctx_vecs = self.seq2vec(attn_oc, opt_lens)
            opt_vecs = self.seq2vec(attn_co, context_lens)

            # logit = self.mlp(vecs).squeeze(-1)
            logit = self.similarity(ctx_vecs, opt_vecs)
            logits.append(logit)

        # pdb.set_trace()
        logits = torch.stack(logits, 1)
        return logits


class RecurrentTransformerEncoder(torch.nn.Module):
    """

    Args:

    """
    def __init__(self, dim_embeddings,
                 n_heads=6, dropout_rate=0.1, dim_ff=128,
                 dim_encoder=102, dim_encoder_ff=256, n_blocks=1):
        super(RecurrentTransformerEncoder, self).__init__()
        self.encoder = TransformerEncoder(
            dim_embeddings,
            dim_encoder,
            n_heads,
            dropout_rate,
            dim_encoder_ff,
            n_blocks=n_blocks
        )
        self.connection = Connection(
            dim_encoder,
            dim_encoder,
            n_heads,
            dropout_rate,
            dim_ff)
        self.register_buffer('padding', torch.zeros(dim_embeddings))
        self.register_buffer('padding2', torch.zeros(dim_encoder))

    def forward(self, seqs, ends):
        first_ends = [min(end[0], 50) for end in ends]
        encoded = self.encoder(seqs[:, :max(first_ends)], first_ends)
        encoded = [seq[:end] for seq, end in zip(encoded, first_ends)]
        history = [[e] for e in encoded]

        context_lens = list(map(len, ends))
        batch_size = seqs.size(0)

        for i in range(0, max(context_lens) - 1):
            workings = list(
                filter(lambda j: context_lens[j] > i + 1,
                       range(batch_size))
            )

            currs = []
            prevs = []
            for working in workings:
                start, end = ends[working][i], ends[working][i + 1]
                if i == 0:
                    curr_len = min(end - start, 150)
                else:
                    curr_len = min(end - start, 100)

                curr = seqs[working, start:start + curr_len]
                currs.append(curr)
                prevs.append(encoded[working])

            curr_lens, currs = pad_seqs(currs, self.padding)
            prev_lens, prevs = pad_seqs(prevs, self.padding2)

            currs = self.encoder(currs, curr_lens)
            outputs = self.connection(prevs, currs, prev_lens)

            for j, working in enumerate(workings):
                start, end = ends[working][i], ends[working][i + 1]
                seq_len = min(end - start, 50)
                encoded[working] = outputs[j][-seq_len:]
                history[working].append(outputs[j][-seq_len:])

        history = [torch.cat(h, 0) for h in history]
        return encoded, history


class MCAN(torch.nn.Module):
    """

    Args:

    """
    def __init__(self, dim_embeddings,
                 dropout_rate=0.0,
                 use_co_att=True,
                 use_intra_att=True,
                 intra_per_utt=False,
                 use_highway_encoder=True,
                 use_projection=True):
        super(MCAN, self).__init__()

        self.use_intra_att = use_intra_att
        self.intra_per_utt = intra_per_utt
        if use_intra_att:
            self.intra_att_encoder = IntraAttention(
                dim_embeddings, use_projection=use_projection
            )

        self.use_co_att = use_co_att
        if use_co_att:
            self.co_att_mean = CoAttention(
                dim_embeddings, pooling='mean',
                use_projection=use_projection
            )
            self.co_att_max = CoAttention(
                dim_embeddings, pooling='max',
                use_projection=use_projection
            )
            self.co_att_align = CoAttention(
                dim_embeddings, pooling='align',
                use_projection=use_projection
            )

        self.use_highway_encoder = use_highway_encoder
        if use_highway_encoder:
            self.highway_encoder = HighwayNetwork(dim_embeddings, n_layers=1)

    def forward(self, context, context_ends, option, option_len):
        context_lens = [ends[-1] for ends in context_ends]
        if self.use_highway_encoder:
            context = self.highway_encoder(context)
            option = self.highway_encoder(option)

        # accumulator for casted features
        ctx_casted_features = []
        opt_casted_features = []

        # intra attention
        padding = torch.zeros_like(context[0, 0])
        if self.use_intra_att:
            if self.intra_per_utt:
                context_intra = []
                for b, ends in enumerate(context_ends):
                    # split to subsequences
                    utts = []
                    for start, end in zip([0] + ends[:-1], ends[1:]):
                        utts.append(list(context[b, start:end]))
                    lens, utts = pad_and_cat(utts, padding)

                    # encode subsequences
                    utt_intras = self.intra_att_encoder(utts, lens)

                    # concatenate back to single sequence
                    utt_intras = [utt_intra[:ul]
                                  for utt_intra, ul in zip(utt_intras, lens)]

                    # add back some padding so shape[0] remain after catted
                    utt_intras.apend(
                        torch.zeros(
                            (context.size(0) - sum(lens), utt_intras.shape[-1])
                        ).to(context.device())
                    )

                    # cat back to a single sequence and accumulate
                    intras = torch.cat(utt_intras, 0)
                    context_intra.append(intras)

                context_intra = torch.stack(context_intra, 0)
            else:
                context_intra = self.intra_att_encoder(context, context_lens)

            option_intra = self.intra_att_encoder(option, option_len)

            ctx_casted_features.append(context_intra)
            opt_casted_features.append(option_intra)

        if self.use_co_att:
            # mean
            context_co, option_co = self.co_att_mean(
                context, context_lens, option, option_len)
            ctx_casted_features.append(context_co)
            opt_casted_features.append(option_co)

            # max
            context_co, option_co = self.co_att_max(
                context, context_lens, option, option_len)
            ctx_casted_features.append(context_co)
            opt_casted_features.append(option_co)

            # align
            context_co, option_co = self.co_att_align(
                context, context_lens, option, option_len)
            ctx_casted_features.append(context_co)
            opt_casted_features.append(option_co)

        ctx_casted_features = torch.cat(ctx_casted_features, dim=-1)
        opt_casted_features = torch.cat(opt_casted_features, dim=-1)
        return ctx_casted_features, opt_casted_features
