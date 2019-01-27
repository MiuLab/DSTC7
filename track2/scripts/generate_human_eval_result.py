#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('responses_dir', type=Path, help='Responses directory.')
    parser.add_argument(
        'model_name_map_path', type=Path, help='Path to model_name_map.csv.')
    args = parser.parse_args()

    return vars(args)


def main(responses_dir, model_name_map_path):
    with model_name_map_path.open() as f:
        reader = csv.DictReader(f)
        model_name_map = [r['model'] for r in reader]

    responses = np.array([])
    for response_path in responses_dir.iterdir():
        with response_path.open() as f:
            reader = csv.reader(f)
            response = [r for r in reader]
        # remove row/col header and convert str to int
        response = [list(map(int, r[1:])) for r in response[1:]]
        responses = np.append(responses, np.array(response).mean(axis=0))
    result = responses.reshape((100, 3, 4))
    mean = result.mean(axis=0)
    std = result.std(axis=0)
    mean = np.concatenate((mean, mean.mean(axis=1, keepdims=True)), axis=1)
    std = np.concatenate((std, std.std(axis=1, keepdims=True)), axis=1)
    questions = ['fluent', 'relevant', 'fact-related', 'informative', 'average']
    result = [{
        'model': model_name_map[i],
        **{q: f'{m:1.2f}({s:1.2f})' for q, m, s in zip(questions, mean[i], std[i])}
    } for i in range(3)]

    output_path = responses_dir.parent / 'results.csv'
    with output_path.open(mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['model'] + questions)
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    import sys
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
