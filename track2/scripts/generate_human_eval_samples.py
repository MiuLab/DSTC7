#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import pickle
import random
import string
from pathlib import Path
from functools import reduce

from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_data_path', metavar='test_data', type=Path, help='Preprocessed test data')
    parser.add_argument(
        'model_output_paths', metavar='model_output', type=Path, nargs='+',
        help='Model output path')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    parser.add_argument(
        '-l', '--min_len', type=int, default=3, help='Min response length')
    parser.add_argument(
        '-p', '--min_overlap', type=int, default=3, help='Min context-fact overlap')
    parser.add_argument(
        '-n', '--n_samples', type=int, default=100, help='Number of samples')
    parser.add_argument(
        '-s', '--random_seed', type=int, default=19, help='Random seed')
    args = parser.parse_args()

    return vars(args)


def load_convs(conv_path):
    with conv_path.open() as f:
        lines = f.readlines()

    convs = {}
    bar = tqdm(lines, desc=f'[*] Loading {conv_path}', leave=False)
    for l in bar:
        _hash, subreddit, conv_id, upvotes, turn_num, context, response = l.split('\t')
        response = response.strip()
        convs[_hash] = {
            # 'subreddit': subreddit,
            # 'conv_id': conv_id,
            # 'upvotes': int(upvotes),
            # 'turn_num': int(turn_num),
            # 'context': context,
            'response': response
        }

    return convs


def generate_samples(test_data_path, model_output_paths, min_len, min_overlap,
                     n_samples):
    data = pickle.load(test_data_path.open(mode='rb'))
    model_name_map = {p.stem: f'M{i+1}' for i, p in enumerate(model_output_paths)}
    model_outputs = {p.stem: load_convs(p) for p in model_output_paths}
    hash_list = reduce(
        set.intersection, [set(o.keys()) for o in model_outputs.values()])
    data = [{
        **c,
        'fact': data['facts'][c['conv_id']][c['fact_rank'][0]]['fact'],
        'responses': {f'{model_name_map[k]}_response': v[c['hash']]['response']
                      for k, v in model_outputs.items()}
    } for c in data['convs'] if c['hash'] in hash_list]

    samples = []
    punc_trans_table = str.maketrans({k: '' for k in string.punctuation})
    for d in data:
        min_response_len = min([len(r.translate(punc_trans_table).split())
                                for r in d['responses'].values()])
        fact_tokens = set(map(str.lower, d['fact']))
        context_tokens = set(map(str.lower, sum(d['context'], [])))
        n_overlaps = \
            len(fact_tokens & context_tokens - set(string.punctuation) - STOP_WORDS)
        if min_response_len > min_len and n_overlaps > min_overlap:
            samples.append({
                'hash': d['hash'],
                'fact': ' '.join(d['fact']),
                'context': 'EOS'.join([' '.join(c) for c in d['context']]),
                **d['responses']
            })

    if n_samples:
        random.shuffle(samples)
        samples = samples[:n_samples]

    return model_name_map, samples


def generate_form(samples):
    form = []
    questions = [
        'Is the response fluent?', 'Is the response relevant and appropriate?',
        'Is the response related to the given fact?',
        'Is the response interesting and informative?']
    for i, sample in enumerate(samples):
        form.append({
            'question_type': 'PAGE_BREAK',
            'question_title': f'Conversation {i+1}',
            'required': 'NO',
        })
        form.append({
            'question_type': 'SECTION_HEADER',
            'question_title': 'Fact',
            'question_help_text': sample['fact'],
            'required': 'NO',
        })
        context = '\n------------------------------\n'.join(
            [c.strip() for c in sample['context'].split('EOS') if c.strip()])
        form.append({
            'question_type': 'SECTION_HEADER',
            'question_title': 'Context',
            'question_help_text': context,
            'required': 'NO',
        })
        for i in range(1, 4):
            form.append({
                'question_type': 'SECTION_HEADER',
                'question_title': f'Response {i}',
                'question_help_text': sample[f'M{i}_response'],
                'required': 'No',
            })
            for q in questions:
                form.append({
                    'question_type': 'SCALE',
                    'question_title': q,
                    'required': 'YES',
                    'choice1': 'Left Label: Strongly Disagree',
                    'choice2': 'Lower Bound: 1',
                    'choice3': 'Right Label: Strongly Agree',
                    'choice4': 'Upper Bound: 5'
                })

    return form


def main(test_data_path, model_output_paths, output_dir, min_len, min_overlap,
         n_samples, random_seed):
    random.seed(random_seed)

    model_name_map, samples = generate_samples(
        test_data_path, model_output_paths, min_len, min_overlap, n_samples)
    form = generate_form(samples)

    model_name_map_path = output_dir / 'model_name_map.csv'
    with model_name_map_path.open(mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'name'])
        writer.writeheader()
        writer.writerows([{'model': k, 'name': v} for k, v in model_name_map.items()])

    output_path = output_dir / 'samples.csv'
    with output_path.open(mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)

    output_path = output_dir / 'form.csv'
    with output_path.open(mode='w') as f:
        fieldnames = [
            'ingested_form_title', 'form_published_url', 'question_type',
            'question_title', 'question_help_text', 'image_url', 'required', 'choice1',
            'choice2', 'choice3', 'choice4']
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(form)


if __name__ == "__main__":
    import sys
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
