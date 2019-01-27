import random
import string
from functools import reduce

import torch
from spacy.lang.en.stop_words import STOP_WORDS
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from constants import SEED


random.seed(SEED)


class SGDataset(Dataset):
    def __init__(self, convs, facts, vocab, topn_facts, use_filter):
        def to_index(sent):
            return {
                'word': [vocab.word.vtoi(w.lower()) for w in sent],
                'orig': [vocab.full.vtoi(w.lower()) for w in sent]
            }

        if use_filter:
            full_conv_count = len(convs)
            filtered_convs = []
            for c in tqdm(convs, desc='[*] Filtering convs', leave=False):
                conv_facts = facts[c['conv_id']]
                relevant_fact_ids = c['fact_rank'][:topn_facts]
                relevant_facts = [conv_facts[i]['fact'] for i in relevant_fact_ids]
                fact_tokens = set(map(str.lower, sum(relevant_facts, [])))
                response_tokens = set(map(str.lower, c['response']))
                intersect_tokens = \
                    fact_tokens & response_tokens - set(string.punctuation) - STOP_WORDS
                if intersect_tokens and len(c['response']) < 20:
                    filtered_convs.append(c)
            filtered_conv_ids = set([c['conv_id'] for c in filtered_convs])
            filtered_facts = {conv_id: facts[conv_id] for conv_id in filtered_conv_ids}

            filtered_conv_count = len(filtered_convs)
            percentage = round(filtered_conv_count / full_conv_count * 100, 3)
            print(f'[#] Filtered data contains {percentage}% '
                  f'({filtered_conv_count}/{full_conv_count}) of full data')

            convs = filtered_convs
            facts = filtered_facts

        self.__convs = []
        for conv in tqdm(convs, desc='[*] Indexizing convs', leave=False):
            conv['context'] = [to_index(c) for c in conv['context']]
            conv['response'] = to_index(conv['response'])
            self.__convs.append(conv)
        convs = None

        self.__facts = {}
        bar = tqdm(facts.items(), desc='[*] Indexizing facts', leave=False)
        for conv_id, fact in bar:
            fact = [{'fact': to_index(f['fact'])} for f in fact]
            self.__facts[conv_id] = fact
        facts = None

        self.__eou_idx = vocab.word.sp.eou.idx
        self.__topn_facts = topn_facts
        self.combine_context = True
        self.combine_facts = True

    def __getitem__(self, index):
        conv = self.__convs[index]
        conv_id = conv['conv_id']

        context = conv['context']
        if self.combine_context:
            context = reduce(
                lambda x, y: {k: x[k] + [self.__eou_idx] + y[k] for k in x}, context)

        facts = self.__facts[conv_id]
        facts = [facts[i]['fact'] for i in conv['fact_rank'][:self.__topn_facts]]
        if self.combine_facts:
            facts = reduce(
                lambda x, y: {k: x[k] + [self.__eou_idx] + y[k] for k in x}, facts)

        response = conv['response']

        return {
            'hash': conv['hash'],
            'context': context,
            'response': response, 'fact': facts
        }

    def __len__(self):
        return len(self.__convs)


def create_collate_fn(vocab, max_seq_len):
    pad_idx = vocab.word.sp.pad.idx
    eos_idx = vocab.word.sp.eos.idx

    def pad(batch, max_seq_len, truncate_mode='start'):
        for i, b in enumerate(batch):
            if truncate_mode == 'start':
                batch[i] = b[:max_seq_len]
            elif truncate_mode == 'end':
                batch[i] = b[-max_seq_len:]
            batch[i] += [pad_idx] * (max_seq_len - len(b))

        return torch.LongTensor(batch)

    def collate_fn(batch):
        context = [b['context']['word'] for b in batch]
        max_context_len = min(max_seq_len, max(map(len, context)))
        padded_context = pad(context, max_context_len, truncate_mode='end')
        context_pad_mask = (padded_context != pad_idx)

        fact = [b['fact']['word'] for b in batch]
        max_fact_len = min(500, max(map(len, fact)))
        padded_fact = pad(fact, max_fact_len)
        fact_pad_mask = (padded_fact != pad_idx)

        response = [b['response']['word'] for b in batch]
        # padded_response = pad(response, max_seq_len)
        padded_response = pad(response, 20)
        pad_for_eos = torch.full((len(padded_response), 1), pad_idx, dtype=torch.long)
        padded_response = torch.cat((padded_response, pad_for_eos), dim=-1)
        # set <eos> token at the end of the sequence
        # [1, 2, 3, 0]    [1, 2, 3, 4] (pad_idx = 0, eos_idx = 4)
        # [1, 0, 0, 0] -> [1, 4, 0, 0]
        # [1, 2, 0, 0]    [1, 2, 4, 0]
        index = torch.arange(len(batch), dtype=torch.long)
        response_len = (padded_response != pad_idx).sum(dim=-1)
        padded_response[index, response_len] = eos_idx
        response_pad_mask = (padded_response != pad_idx)

        orig_response = [b['response']['orig'] for b in batch]

        return {
            'hash': [b['hash'] for b in batch],
            'context': padded_context,
            'fact': padded_fact,
            'response': padded_response,
            'context_pad_mask': context_pad_mask,
            'fact_pad_mask': fact_pad_mask,
            'response_pad_mask': response_pad_mask,
            'orig_response': orig_response
        }

    return collate_fn


def create_data_loader(convs, facts, vocab, topn_facts, use_filter, max_seq_len,
                       batch_size, n_workers, drop_last, shuffle=True, **_):
    dataset = SGDataset(convs, facts, vocab, topn_facts, use_filter)
    collate_fn = create_collate_fn(vocab, max_seq_len)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
        collate_fn=collate_fn, drop_last=drop_last)

    return data_loader
