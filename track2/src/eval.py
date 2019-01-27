import argparse
import csv
import math
import sys
import ipdb

from collections import Counter
from nlgeval import compute_metrics
from nltk.translate.nist_score import corpus_nist
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--ref', dest='ref_path',
        type=Path, help='the path of reference data')
    parser.add_argument(
        '-f', '--hyp', dest='hyp_paths', nargs='+',
        type=Path, help='the path of hypothesis data')
    parser.add_argument(
        '-o', '--out', dest='output_dir',
        type=Path, help='directory to put result.csv')
    args = parser.parse_args()
    return vars(args)


def diversity(corpus, n):
    c = Counter([
        ' '.join(sent[idx:idx+n]) for sent in corpus
        for idx in range(len(sent)-n)])
    numerator = len(c)
    denominator = sum([len(sent) for sent in corpus])
    return numerator / denominator


def entropy(corpus, n):
    c = Counter([
        ' '.join(sent[idx:idx+n]) for sent in corpus
        for idx in range(len(sent)-n)])
    freqs = [v for k, v in c.most_common()]
    sum_freq = sum(freqs)

    return sum((-1 / sum_freq) * (f * math.log(f / sum_freq)) for f in freqs)


def eval(ref_path, hyp_path):
    print(f'Evaluating {hyp_path} ...')
    with open(ref_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        ref_data = {line[0]: line[-1] for line in reader}
    with open(hyp_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        hyp_data = {line[0]: line[-1] for line in reader}
    ref_id = set(ref_data)
    hyp_id = set(hyp_data)
    inter_id = ref_id & hyp_id
    data = {}
    for ID in inter_id:
        data[ID] = [ref_data[ID], hyp_data[ID]]

    ref_file = open(f"{ref_path}.plain", 'w')
    hyp_file = open(f"{hyp_path}.plain", 'w')
    for _, d in data.items():
        ref_file.write(f"{d[0]}\n")
        hyp_file.write(f"{d[1]}\n")
    ref_file.close()
    hyp_file.close()

    r = [[d[0].split(' ')] for _, d in data.items()]
    h = [d[1].split(' ') for _, d in data.items()]

    metrics_dict = compute_metrics(
        hypothesis=f"{hyp_path}.plain",
        references=[f"{ref_path}.plain"],
        no_skipthoughts=True,
        no_glove=True)

    for n in range(1, 3):
        metrics_dict[f'Nist_{n}'] = corpus_nist(r, h, n=n)
    for n in range(1, 3):
        metrics_dict[f'Div_{n}'] = diversity(h, n=n)
    for n in range(1, 5):
        metrics_dict[f'Ent_{n}'] = entropy(h, n=n)

    print(f"Nist_1: {corpus_nist(r, h, n=1):.6f}")
    print(f"Nist_2: {corpus_nist(r, h, n=2):.6f}")
    # print(f"Nist_3: {corpus_nist(r, h, n=3):.6f}")
    # print(f"Nist_4: {corpus_nist(r, h, n=4):.6f}")

    print(f"Div_1: {diversity(h, n=1):.6f}")
    print(f"Div_2: {diversity(h, n=2):.6f}")

    print(f"Entropy_1: {entropy(h, n=1):.6f}")
    print(f"Entropy_2: {entropy(h, n=2):.6f}")
    print(f"Entropy_3: {entropy(h, n=3):.6f}")
    print(f"Entropy_4: {entropy(h, n=4):.6f}")

    metrics_dict['Model'] = Path(hyp_path).stem

    return metrics_dict


def main(ref_path, hyp_paths, output_dir):
    result = [eval(ref_path, hyp_path) for hyp_path in hyp_paths]
    output_path = output_dir / 'result.scv'
    with output_path.open(mode='w') as f:
        fieldnames = list(result[0].keys())
        fieldnames = ['Model'] + [fn for fn in fieldnames if fn != 'Model']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
