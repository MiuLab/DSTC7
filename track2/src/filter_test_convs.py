#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import ipdb
from tqdm import tqdm


undisclosed_str = '__UNDISCLOSED__'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hash_list_path', type=Path, help='Testing data hash list path')
    parser.add_argument('conv_path', type=Path, help='Input conv path')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    args = parser.parse_args()

    return vars(args)


def main(hash_list_path, conv_path, output_dir):
    test_conv_path = output_dir / 'test.convos.txt'
    if test_conv_path.exists():
        print(f'[!] {test_conv_path} already exists')
        exit(1)

    print('[*] Filtering test convs')
    with hash_list_path.open() as f:
        hash_list = [l.strip() for l in f.readlines()]
    print(f'[#] Hash count: {len(hash_list)}')
    with conv_path.open() as f:
        lines = f.readlines()
    lines = [l for l in tqdm(lines, desc='[*] Filtering', leave=False)
             if l.strip().split('\t')[0] in hash_list]
    lines = list(set(lines))
    print(f'[#] Filtered convs count: {len(lines)}')
    with test_conv_path.open(mode='w') as f:
        for l in lines:
            f.write(l)
    print(f'[-] Filtered test convs saved to {test_conv_path}')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
