import pdb
import random
import torch
from torch.utils.data import Dataset


class DSTC7Dataset(Dataset):
    def __init__(self, data, padding=0,
                 n_negative=4, n_positive=1,
                 context_padded_len=400, option_padded_len=50,
                 min_context_len=4):
        self.data = data
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.context_padded_len = context_padded_len
        self.option_padded_len = option_padded_len
        self.padding = padding
        self.min_context_len = min_context_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        if self.min_context_len >= len(data['utterance_ends']):
            min_context_len = len(data['utterance_ends'])
        else:
            min_context_len = self.min_context_len

        context_len = random.randint(
            min_context_len,
            len(data['utterance_ends'])
        )
        if context_len < len(data['utterance_ends']):
            context_end = data['utterance_ends'][context_len - 1]
            next_end = data['utterance_ends'][context_len]
            data['options'][0] = data['context'][context_end:next_end]
            data['utterance_ends'] = data['utterance_ends'][:context_len]

        answers = data['options'][:data['n_corrects']]
        for ans in answers:
            try:
                ans_index = data['options'][data['n_corrects']:].index(ans) \
                    + data['n_corrects']
                del data['options'][ans_index]
            except ValueError:
                pass

        if self.n_negative == self.n_positive == -1:
            n_positive = data['n_corrects']
            n_negative = 100 - n_positive
        else:
            n_positive = min(data['n_corrects'], self.n_positive)
            n_negative = min(self.n_negative + self.n_positive - n_positive,
                             len(data['options']) - n_positive)

        # sample positive indices
        positive_indices = list(range(data['n_corrects']))
        random.shuffle(positive_indices)
        positive_indices = positive_indices[:n_positive]

        # sample negative indices
        negative_indices = list(range(data['n_corrects'],
                                      len(data['options'])))
        if data['n_corrects'] > 0:
            random.shuffle(negative_indices)
        negative_indices = negative_indices[:n_negative]

        data['options'] = (
            [data['options'][i] for i in positive_indices]
            + [data['options'][i] for i in negative_indices]
        )
        data['option_suggested'] = (
            [data['option_suggested'][i] for i in positive_indices]
            + [data['option_suggested'][i] for i in negative_indices]
        )
        data['option_prior'] = (
            [data['option_prior'][i] for i in positive_indices]
            + [data['option_prior'][i] for i in negative_indices]
        )

        data['labels'] = [1] * n_positive + [0] * n_negative

        return data

    def collate_fn(self, datas):
        batch = {}

        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['utterance_ends'] = [
            list(filter(lambda e: e < self.context_padded_len,
                        data['utterance_ends']))
            for data in datas]
        for end in batch['utterance_ends']:
            if len(end) == 0:
                end.append(40)

        batch['labels'] = torch.tensor([data['labels'] for data in datas])

        # build tensor of context
        batch['context_lens'] = [len(data['context']) for data in datas]
        padded_len = min(self.context_padded_len, max(batch['context_lens']))
        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas]
        )

        # build tensor of options
        batch['option_lens'] = [[max(len(opt), 1) for opt in data['options']]
                                for data in datas]
        batch['options'] = torch.tensor(
            [[pad_to_len(opt, self.option_padded_len, self.padding)
              for opt in data['options']]
             for data in datas]
        )

        if 'prior' in datas[0]:
            batch['prior'] = torch.tensor(
                [pad_to_len(data['prior'], padded_len, self.padding)
                 for data in datas]
            ).float()
            batch['suggested'] = torch.tensor(
                [pad_to_len(data['suggested'], padded_len, self.padding)
                 for data in datas]
            ).float()
            batch['option_prior'] = torch.tensor(
                [[pad_to_len(opt, self.option_padded_len, self.padding)
                  for opt in data['option_prior']]
                 for data in datas]
            ).float()
            batch['option_suggested'] = torch.tensor(
                [[pad_to_len(opt, self.option_padded_len, self.padding)
                  for opt in data['option_suggested']]
                 for data in datas]
            ).float()

        if 'option_ids' in datas[0]:
            batch['option_ids'] = [data['option_ids'] for data in datas]

        return batch

    def collate_fn_role(self, datas):
        batch = {}

        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['utterance_ends'] = [
            list(filter(lambda e: e < self.context_padded_len,
                        data['utterance_ends']))
            for data in datas]
        for end in batch['utterance_ends']:
            if len(end) == 0:
                end.append(40)

        for speaker in [1, 2]:
            batch['utterance_ends{}'.format(speaker)] = [
                list(filter(lambda e: e < self.context_padded_len,
                            data['utterance_ends{}'.format(speaker)]))
                for data in datas]
            for end in batch['utterance_ends{}'.format(speaker)]:
                if len(end) == 0:
                    end.append(40)

            batch['utterance_ends{}_masked'.format(speaker)] = [
                list(filter(lambda e: e < self.context_padded_len,
                            data['utterance_ends{}_masked'.format(speaker)]))
                for data in datas]
            for end in batch['utterance_ends{}_masked'.format(speaker)]:
                if len(end) == 0:
                    end.append(40)

        batch['labels'] = torch.tensor([data['labels'] for data in datas])

        for speaker in [1, 2]:
            batch['context_lens{}_masked'.format(speaker)] = [
                len(data['context{}_masked'.format(speaker)]) for data in datas
            ]
            padded_len = min(
                self.context_padded_len,
                max(batch['context_lens{}_masked'.format(speaker)])
            )
            batch['context{}_masked'.format(speaker)] = torch.tensor(
                [pad_to_len(
                    data['context{}_masked'.format(speaker)],
                    padded_len, self.padding
                 )
                 for data in datas]
            )

        for speaker in [1, 2]:
            batch['context_lens{}'.format(speaker)] = [
                len(data['context{}'.format(speaker)]) for data in datas
            ]
            padded_len = min(
                self.context_padded_len,
                max(batch['context_lens{}'.format(speaker)])
            )
            batch['context{}'.format(speaker)] = torch.tensor(
                [pad_to_len(
                    data['context{}'.format(speaker)],
                    padded_len, self.padding
                 )
                 for data in datas]
            )

        # build tensor of context
        batch['context_lens'] = [len(data['context']) for data in datas]
        padded_len = min(self.context_padded_len, max(batch['context_lens']))
        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas]
        )

        # build tensor of options
        batch['option_lens'] = [[max(len(opt), 1) for opt in data['options']]
                                for data in datas]
        batch['options'] = torch.tensor(
            [[pad_to_len(opt, self.option_padded_len, self.padding)
              for opt in data['options']]
             for data in datas]
        )

        if 'prior' in datas[0]:
            batch['prior'] = torch.tensor(
                [pad_to_len(data['prior'], padded_len, self.padding)
                 for data in datas]
            ).float()
            batch['suggested'] = torch.tensor(
                [pad_to_len(data['suggested'], padded_len, self.padding)
                 for data in datas]
            ).float()
            batch['option_prior'] = torch.tensor(
                [[pad_to_len(opt, self.option_padded_len, self.padding)
                  for opt in data['option_prior']]
                 for data in datas]
            ).float()
            batch['option_suggested'] = torch.tensor(
                [[pad_to_len(opt, self.option_padded_len, self.padding)
                  for opt in data['option_suggested']]
                 for data in datas]
            ).float()

        if 'option_ids' in datas[0]:
            batch['option_ids'] = [data['option_ids'] for data in datas]

        return batch


def pad_to_len(arr, padded_len, padding=0):
    padded = [padding] * padded_len
    n_copy = min(len(arr), padded_len)
    padded[:n_copy] = arr[:n_copy]
    return padded
