#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import sys
from pathlib import Path

import ipdb
import numpy as np
import ruamel_yaml
import torch
from tqdm import tqdm

from box import Box
from constants import SEED, DEVICE
from dataset import create_data_loader
from model import Model
from vocab import create_vocab


np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path, help='Data path')
    parser.add_argument('orig_convos_path', type=Path, help='Original convos.txt')
    parser.add_argument('config_path', type=Path, help='Config path')
    parser.add_argument('model_path', type=Path, help='Model path')
    parser.add_argument('output_path', type=Path, help='Prediction output path')
    args = parser.parse_args()

    return vars(args)


def main(data_path, orig_convos_path, config_path, model_path, output_path):
    with config_path.open() as f:
        cfg = Box(ruamel_yaml.safe_load(f))
    print(f'[-] Config loaded from {config_path}')
    cfg.data.use_filter = False
    cfg.data.drop_last = False

    with data_path.open(mode='rb') as f:
        data = pickle.load(f)
        conv = data['convs']
        fact = data['facts']
        print(f'[-] Preprocessed test data loaded from {data_path}')

    vocab = create_vocab([], {}, [], {}, **cfg.vocab)

    print('[*] Creating test data loader')
    data_loader = create_data_loader(conv, fact, vocab, **cfg.data)
    print('[-] Test data loader created')

    # the +1 is for <eos> token
    # cfg.net.max_de_seq_len = cfg.data.max_seq_len + 1
    cfg.net.max_de_seq_len = 20 + 1
    model = Model(vocab, cfg.net, cfg.optim)
    model.load_state(model_path)

    predictions = infer(vocab, data_loader, model)
    save_predictions(orig_convos_path, predictions, output_path)


def infer(vocab, data_loader, model):
    model.set_eval()
    torch.set_grad_enabled(False)

    predictions = {}
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'[Test]', leave=False)):
        _hash = batch['hash']
        context = batch['context'].to(device=DEVICE)
        fact = batch['fact'].to(device=DEVICE)
        context_pad_mask = batch['context_pad_mask'].to(device=DEVICE)
        fact_pad_mask = batch['fact_pad_mask'].to(device=DEVICE)

        prediction, *_ = model.infer(context, fact, context_pad_mask, fact_pad_mask, 8)
        for h, p in zip(_hash, prediction):
            p = [w for w in p if w not in
                 [vocab.word.sp.unk.idx, vocab.word.sp.pad.idx]]
            try:
                eos_idx = p.index(vocab.word.sp.eos.idx)
            except ValueError:
                pass
            else:
                p = p[:eos_idx]
            p = ' '.join([vocab.word.itov(i) for i in p])
            predictions[h] = p

    return predictions


def save_predictions(orig_convos_path, predictions, output_path):
    with orig_convos_path.open() as f:
        lines = f.readlines()
    with output_path.open(mode='w') as f:
        for l in lines:
            ll = l.strip().split('\t')
            _hash = ll[0]
            if _hash in predictions:
                ll[-1] = predictions[_hash]
                f.write('\t'.join(ll) + '\n')
    print(f'[-] Output saved to {output_path}')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
