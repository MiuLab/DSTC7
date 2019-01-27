#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path

import ipdb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path, help='Data directory.')
    parser.add_argument('output_path', type=Path, help='Corpus output path.')
    args = parser.parse_args()

    return vars(args)


def load_convs(conv_path):
    with conv_path.open(encoding='utf8') as f:
        lines = f.readlines()

    convs = set()
    split_pattern = re.compile('START|EOS')
    bar = tqdm(lines, desc=f'[*] Loading {conv_path}', leave=False)
    for l in bar:
        *_, context, response = l.strip().split('\t')
        for c in split_pattern.split(context):
            if c.strip():
                convs.add(c.strip())
        convs.add(response)

    return convs


def load_facts(fact_path):
    with fact_path.open() as f:
        lines = f.readlines()

    facts = set()
    bar = tqdm(lines, desc=f'[*] Loading {fact_path}', leave=False)
    for l in bar:
        *_, fact = l.strip().split('\t')
        if fact.strip():
            facts.add(fact)

    return facts


def main(data_dir, output_path):
    if output_path.exists():
        print(f'[!] {output_path} already exists')
        exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = []
    corpus += load_convs(data_dir / 'train.convos.txt')
    corpus += load_facts(data_dir / 'train.facts.txt')
    corpus += load_facts(data_dir / 'dev.convos.txt')
    corpus += load_convs(data_dir / 'dev.facts.txt')

    with output_path.open(mode='w', encoding='utf8') as f:
        for line in tqdm(corpus, desc='Writing to output', leave=False):
            f.write(line + '\n')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
