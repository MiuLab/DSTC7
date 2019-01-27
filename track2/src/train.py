#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import sys
from pathlib import Path

import ipdb
import ruamel_yaml

from box import Box
from dataset import create_data_loader
from model import Model
from trainer import Trainer
from vocab import create_vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='Config path')
    args = parser.parse_args()

    return vars(args)


def main(config_path):
    with config_path.open() as f:
        cfg = Box(ruamel_yaml.safe_load(f))
    print(f'[-] Config loaded from {config_path}')
    print(f'[-] Experiment: {cfg.train.exp_name}')

    print('[*] Loading preprocessed data')
    preprocessed_dir = Path(cfg.data.preprocessed_dir)
    train_data_path = preprocessed_dir / 'train.pkl'
    with train_data_path.open(mode='rb') as f:
        train_data = pickle.load(f)
        train_conv = train_data['convs']
        train_fact = train_data['facts']
        print(f'[-] Preprocessed train data loaded from {train_data_path}')
    dev_data_path = preprocessed_dir / 'dev.pkl'
    with dev_data_path.open(mode='rb') as f:
        dev_data = pickle.load(f)
        dev_conv = dev_data['convs']
        dev_fact = dev_data['facts']
        print(f'[-] Preprocessed dev data loaded from {dev_data_path}')

    vocab = create_vocab(train_conv, train_fact, dev_conv, dev_fact, **cfg.vocab)

    print('[*] Creating train data loaders')
    train_data_loader = create_data_loader(train_conv, train_fact, vocab, **cfg.data)
    print('[-] Train data loader created')
    print('[*] Creating dev data loaders')
    dev_data_loader = create_data_loader(dev_conv, dev_fact, vocab, **cfg.data)
    print('[-] Dev data loader created')

    # the +1 is for <eos> token
    # cfg.net.max_de_seq_len = cfg.data.max_seq_len + 1
    cfg.net.max_de_seq_len = 20 + 1
    model = Model(vocab, cfg.net, cfg.optim)

    print('[-] Start training')
    trainer = Trainer(
        config_path, cfg.train, vocab, train_data_loader, dev_data_loader, model)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
