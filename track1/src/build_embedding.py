import argparse
import json
import logging
import os
import pickle
import spacy
import copy
import sys
import pdb
# from IPython import embed
import traceback
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from embeddings import Embeddings


def collect_words(train_path, valid_path, n_workers=16):
    logging.info('Loading valid data...')
    with open(valid_path) as f:
        valid = json.load(f)
    logging.info('Tokenize words in valid...')
    valid = tokenize_data_parallel(valid, args.n_workers)

    logging.info('Loading train data...')
    with open(train_path) as f:
        train = json.load(f)
    logging.info('Tokenize words in train...')
    train = tokenize_data_parallel(train, args.n_workers)

    logging.info('Building word list...')
    words = {}
    data = train + valid
    for sample in tqdm(data):
        utterances = (
            [message['utterance']
             for message in sample['messages-so-far']]
            + [option['utterance']
               for option in sample['options-for-correct-answers']]
            + [option['utterance']
               for option in sample['options-for-next']]
        )

        for utterance in utterances:
            for word in utterance:
                word = word.lower()
                if word not in words:
                    words[word] = 0
                else:
                    words[word] += 1

    return words


def tokenize_data_parallel(data, n_workers=16):
    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(data) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(data) - 1
            else:
                batch_end = (len(data) // n_workers) * (i + 1)

            batch = data[batch_start: batch_end]
            results[i] = pool.apply_async(tokenize_data, [batch])

        pool.close()
        pool.join()

    data = []
    for result in results:
        data += result.get()

    return data


def tokenize_data(data):
    nlp = spacy.load('en_core_web_sm',
                     disable=['tagger', 'ner', 'parser', 'textcat'])

    def tokenize(text):
        return [token.text
                for token in nlp(text)]

    data = copy.deepcopy(data)
    for sample in data:
        for i, message in enumerate(sample['messages-so-far']):
            sample['messages-so-far'][i]['utterance'] = \
                tokenize(message['utterance'])
        for i, message in enumerate(sample['options-for-correct-answers']):
            sample['options-for-correct-answers'][i]['utterance'] = \
                tokenize(message['utterance'])
        for i, message in enumerate(sample['options-for-next']):
            sample['options-for-next'][i]['utterance'] = \
                tokenize(message['utterance'])

    return data


def oov_statistics(words, word_dict):
    total_word = 0
    oov = {}
    for word, count in words.items():
        total_word += count
        if word not in word_dict:
            oov[word] = count

    oov_counts = np.array([v for v in oov.values()])
    oov_counts = np.sort(oov_counts)
    counts_cum_sum = (
        np.cumsum(oov_counts[::-1]) + len(word_dict)) / total_word
    return oov, counts_cum_sum


def main(args):
    logging.info('Collecting words...')
    words = collect_words(args.train_path, args.valid_path)

    logging.info('Building embeddings...')
    embeddings = Embeddings(args.embedding_path, list(words.keys()),
                            not args.keep_oov)

    embeddings.add('speaker1')
    embeddings.add('speaker2')

    logging.info('Saving embedding to {}'.format(args.output))
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)

    if args.words is not None:
        with open(args.words, 'wb') as f:
            pickle.dump(words, f)
    """
    logging.info('Calculating OOV statics...')
    oov, cum_sum = oov_statistics(words, embeddings.word_dict)
    logging.info('There are {} OOVS'.format(cum_sum[-1]))
    """

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Build embedding pickle by extracting vector from'
                    ' pretrained embeddings only for words in the data.')
    parser.add_argument('train_path', type=str,
                        help='[input] Path to the train data.')
    parser.add_argument('valid_path', type=str,
                        help='[input] Path to the dev data.')
    parser.add_argument('embedding_path', type=str,
                        help='[input] Path to the embedding .vec file (such as'
                             'FastText or Glove).')
    parser.add_argument('output', type=str,
                        help='[output] Path to the output pickle file.')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--words', type=str, default=None,
                        help='If a path is specified, list of words in the'
                             'data will be dumped.')
    parser.add_argument('--keep_oov', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
