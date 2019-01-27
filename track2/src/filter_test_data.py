#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import re
import sys
from collections import defaultdict, Counter
from functools import partial
from pathlib import Path

import faiss
import ipdb
import numpy as np
from tqdm import tqdm


undisclosed_str = '__UNDISCLOSED__'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hash_list_path', type=Path, help='Testing data hash list path')
    parser.add_argument('conv_path', type=Path, help='Input conv path')
    parser.add_argument('fact_path', type=Path, help='Input fact path')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    args = parser.parse_args()

    return vars(args)


def tokenize(sent):
    return sent.split()


def load_convs(conv_path, hash_list_path):
    with hash_list_path.open() as f:
        hash_list = [l.strip() for l in f.readlines()]

    with conv_path.open() as f:
        lines = f.readlines()
    convs = []
    split_pattern = re.compile('START|EOS')
    bar = tqdm(lines, desc=f'[*] Loading {conv_path}', leave=False)
    for l in bar:
        hash_, subreddit, conv_id, upvotes, turn_num, context, response = \
            l.strip().split('\t')
        context = [c.strip() for c in split_pattern.split(context) if c.strip()]
        if context and hash_ in hash_list:
            convs.append({
                'hash': hash_,
                'subreddit': subreddit,
                'conv_id': conv_id,
                'upvotes': int(upvotes),
                'turn_num': int(turn_num),
                'context': [tokenize(c) for c in context],
                'response': [undisclosed_str]
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


def process_data(hash_list_path, conv_path, fact_path):
    convs = load_convs(conv_path, hash_list_path)
    facts = load_facts(fact_path)

    inter_ids = set([c['conv_id'] for c in convs]) & set(facts.keys())
    convs = [c for c in convs if c['conv_id'] in inter_ids]
    facts = {k: v for k, v in facts.items() if k in inter_ids}

    print(f'[#] Number of convs: {len(convs)}')

    def tfidf(vi, idf, doc, norm=False):
        tf = Counter(doc)
        x = np.zeros(len(vi))
        for t, f in tf.items():
            x[vi[t]] = f
        tf = np.array(x)
        tfidf = tf * idf
        if norm:
            tfidf /= np.linalg.norm(tfidf)

        return tfidf.astype(np.float32)

    bar = tqdm(convs, desc='[*] Sorting relevant facts', leave=False)
    for i, conv in enumerate(bar):
        context = [[w.lower() for w in c] for c in conv['context']]
        fact = [[w.lower() for w in f['fact']] for f in facts[conv['conv_id']]]
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


def main(hash_list_path, conv_path, fact_path, output_dir):
    test_path = output_dir / 'test.pkl'
    if test_path.exists():
        print(f'[!] {test_path} already exists, skipped')
    else:
        print('[*] Processing test data')
        test_data = process_data(hash_list_path, conv_path, fact_path)
        with test_path.open(mode='wb') as f:
            pickle.dump(test_data, f, protocol=4)
        print(f'[-] Preprocessed test data saved to {test_path}')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
