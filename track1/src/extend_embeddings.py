import argparse
import json
import logging
import pickle
import sys
import pdb
# from IPython import embed
import traceback
from tqdm import tqdm
from IPython import embed
from build_embedding import tokenize_data_parallel, oov_statistics


def collect_words(data_path, n_workers=16):
    logging.info('Loading data...')
    with open(data_path) as f:
        data = json.load(f)
    logging.info('Tokenize words in data...')
    data = tokenize_data_parallel(data, args.n_workers)

    logging.info('Building word list...')
    words = {}
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


def main(args):
    logging.info('Loading embeddings')
    with open(args.origin_path, 'rb') as f:
        embeddings = pickle.load(f)

    logging.info('Collecting words')
    words = collect_words(args.data_path)

    logging.info('Extending embeddings...')
    embeddings.extend(args.embedding_path, list(words.keys()),
                      not args.keep_oov)

    logging.info('Saving embedding to {}'.format(args.output))
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)

    if args.words is not None:
        with open(args.words, 'wb') as f:
            pickle.dump(words, f)

    # logging.info('Calculating OOV statics...')
    # oov, cum_sum = oov_statistics(words, embeddings.word_dict)
    # logging.info('There are {} OOVS'.format(cum_sum[-1]))
    # embed()


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Build embedding pickle by extracting vector from'
                    ' pretrained embeddings only for words in the data.')
    parser.add_argument('origin_path', type=str,
                        help='[input] Path to the original embeddings.')
    parser.add_argument('data_path', type=str,
                        help='[input] Path to the data.')
    parser.add_argument('embedding_path', type=str,
                        help='[output] Path to the embedding .vec file (such'
                             'as FastText or Glove).')
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
