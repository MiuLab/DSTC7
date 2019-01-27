#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import re
import string
import sys
from collections import defaultdict, Counter
from functools import partial
from pathlib import Path

import ipdb
import faiss
import numpy as np
import ruamel_yaml
from box import Box
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='Config path')
    args = parser.parse_args()

    return vars(args)


def tokenize(sent):
    return sent.split()


def load_convs(conv_path):
    with conv_path.open() as f:
        lines = f.readlines()

    convs = []
    split_pattern = re.compile('START|EOS')
    bar = tqdm(lines, desc=f'[*] Loading {conv_path}', leave=False)
    for l in bar:
        hash_, subreddit, conv_id, upvotes, turn_num, context, response = \
            l.strip().split('\t')
        context = [c.strip() for c in split_pattern.split(context) if c.strip()]
        if context:
            convs.append({
                'hash': hash_,
                'subreddit': subreddit,
                'conv_id': conv_id,
                'upvotes': int(upvotes),
                'turn_num': int(turn_num),
                'context': [tokenize(c) for c in context],
                'response': tokenize(response)
            })

    return convs


def load_facts(fact_path):
    with fact_path.open() as f:
        lines = f.readlines()

    facts = defaultdict(list)
    bar = tqdm(lines, desc=f'[*] Loading {fact_path}', leave=False)
    for l in bar:
        hash_, subreddit, conv_id, domain, fact = l.split('\t')
        if fact.strip():
            facts[conv_id].append({
                'hash': hash_,
                'subreddit': subreddit,
                'domain': domain,
                'fact': tokenize(fact.strip())
            })

    return facts


def process_data(mode, cfg):
    convs = []
    facts = {}
    data_dir = Path(cfg.data_dir)
    for year_month in cfg[mode]:
        convs += load_convs(data_dir / mode / f'{year_month}.convos.txt')
        facts.update(load_facts(data_dir / mode / f'{year_month}.facts.txt'))

    inter_ids = set([c['conv_id'] for c in convs]) & set(facts.keys())
    convs = [c for c in convs if c['conv_id'] in inter_ids]
    facts = {k: v for k, v in facts.items() if k in inter_ids}

    def tfidf(vi, idf, doc, norm=False):
        tf = Counter(doc)
        x = np.zeros(len(vi))
        for t, f in tf.items():
            x[vi[t]] = f
        tf = np.array(x)
        tfidf = np.log(1 + tf) * np.log(1 + idf)
        if norm:
            tfidf /= np.linalg.norm(tfidf)

        return tfidf.astype(np.float32)

    stop_punc = STOP_WORDS | set(string.punctuation)
    bar = tqdm(convs, desc='[*] Sorting relevant facts', leave=False)
    for i, conv in enumerate(bar):
        context = [[w.lower() for w in c if w not in stop_punc]
                   for c in conv['context']]
        fact = [[w.lower() for w in f['fact'] if w not in stop_punc]
                for f in facts[conv['conv_id']]]
        docs = context + fact

        iv = list(set(sum(docs, [])))
        vi = {v: i for i, v in enumerate(iv)}

        df = Counter(sum([list(set(d)) for d in docs], []))
        df = [df[iv[i]] for i in range(len(iv))]
        idf = len(docs) / np.array(df)

        _tfidf = partial(tfidf, vi, idf)

        index = faiss.IndexFlatIP(len(iv))
        index.add(np.array([_tfidf(f) for f in fact]))
        query = np.array([_tfidf(sum(context, []))])
        _, indices = index.search(query, 5)
        convs[i]['fact_rank'] = indices[0]

    return {'convs': convs, 'facts': facts}


def main(config_path):
    with config_path.open() as f:
        cfg = Box(ruamel_yaml.safe_load(f))
    print(f'[-] Config loaded from {config_path}')

    preprocessed_dir = Path(cfg.preprocessed_dir)
    if not preprocessed_dir.exists():
        preprocessed_dir.mkdir()
        print(f'[-] Preprocessed data dir created at {preprocessed_dir}')

    train_path = preprocessed_dir / 'train.pkl'
    if train_path.exists():
        print(f'[!] {train_path} already exists, skipped')
    else:
        print('[*] Processing train data')
        train_data = process_data('train', cfg)
        with train_path.open(mode='wb') as f:
            pickle.dump(train_data, f, protocol=4)
        print(f'[-] Preprocessed train data saved to {train_path}')

    dev_path = preprocessed_dir / 'dev.pkl'
    if dev_path.exists():
        print(f'[!] {dev_path} already exists, skipped')
    else:
        print('[*] Processing dev data')
        dev_data = process_data('dev', cfg)
        with dev_path.open(mode='wb') as f:
            pickle.dump(dev_data, f, protocol=4)
        print(f'[-] Preprocessed dev data saved to {dev_path}')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
