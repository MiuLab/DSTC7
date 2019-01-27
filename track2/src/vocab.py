import string
from collections import namedtuple, Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


SpecialToken = namedtuple('SpecialToken', ['sym', 'idx'])


class SpecialVocab:
    def __init__(self, special_tokens):
        self.__special_tokens = special_tokens
        for i, tok in enumerate(special_tokens):
            setattr(self, tok, SpecialToken(sym=f'<{tok}>', idx=i))

    def __len__(self):
        return len(self.__special_tokens)

    def __iter__(self):
        self.__iter_idx = 0
        return self

    def __next__(self):
        if self.__iter_idx < len(self):
            self.__iter_idx += 1
            return getattr(self, self.__special_tokens[self.__iter_idx - 1])
        raise StopIteration


class Vocab:
    def __init__(self, special_tokens, word_frequency=None, glove_path=None, size=None,
                 emb_dim=None):
        self.__special = SpecialVocab(special_tokens)
        if glove_path:
            with glove_path.open() as f:
                lines = f.readlines()
            if not size:
                size = len(lines)
            self.__emb_dim = len(lines[0].strip().split()) - 1
            self.__ie = np.zeros((len(self.__special) + size, self.__emb_dim))

            self.__iv = [v.sym for v in self.__special]
            for v in self.__special:
                if v.sym != '<pad>':
                    self.__ie[v.idx] = np.random.normal(size=self.__emb_dim)
            bar = tqdm(lines[:size], desc='[*] Loading GloVe', leave=False)
            for i, l in enumerate(bar):
                v, *e = l.replace(u'\xa0', '<space>').strip().split()
                self.__iv.append(v)
                self.__ie[len(self.__special) + i] = list(map(float, e))
            self.__vi = {v: i for i, v in enumerate(self.__iv)}
            if emb_dim:
                self.__emb_dim = emb_dim
        else:
            self.__iv = [v.sym for v in self.__special] + \
                [v for v, _ in word_frequency.most_common(size)]
            self.__vi = {v: i for i, v in enumerate(self.__iv)}
            self.__emb_dim = emb_dim

    def vtoi(self, v):
        return self.__vi.get(v, self.__special.unk.idx)

    def itov(self, i):
        return self.__iv[i]

    @property
    def emb_dim(self):
        return self.__emb_dim

    @property
    def emb(self):
        return self.__ie

    @property
    def sp(self):
        return self.__special

    @property
    def n_special(self):
        return len(self.__special)

    def __len__(self):
        return len(self.__vi)


def create_vocab(train_conv, train_fact, dev_conv, dev_fact, glove_path,
                 word_special_tokens, word_vocab_size, word_emb_dim,
                 char_special_tokens, char_emb_dim, full_special_tokens):
    print('[*] Creating vocab')
    word_freq = Counter()
    bar = tqdm(
        train_conv + dev_conv, desc='[*] Collecting tokens from convs', leave=False)
    for conv in bar:
        for c in conv['context']:
            word_freq.update([w.lower() for w in c])
        word_freq.update([w.lower() for w in conv['response']])
    bar = tqdm(
        train_fact.values(), desc='[*] Collecting tokens from train facts', leave=False)
    for fact in bar:
        for f in fact:
            word_freq.update([w.lower() for w in f['fact']])
    bar = tqdm(
        dev_fact.values(), desc='[*] Collecting tokens from dev facts', leave=False)
    for fact in bar:
        for f in fact:
            word_freq.update([w.lower() for w in f['fact']])

    if glove_path:
        word_vocab = Vocab(
            word_special_tokens, glove_path=Path(glove_path), size=word_vocab_size)
    else:
        word_vocab = Vocab(
            word_special_tokens, word_freq, size=word_vocab_size, emb_dim=word_emb_dim)
    char_vocab = Vocab(
        char_special_tokens, Counter(list(string.printable)), emb_dim=char_emb_dim)
    full_vocab = Vocab(full_special_tokens, word_freq)
    VocabTuple = namedtuple('Vocab', ['word', 'char', 'full'])
    vocab = VocabTuple(word=word_vocab, char=char_vocab, full=full_vocab)
    print('[-] Vocab created')

    return vocab
