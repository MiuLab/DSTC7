import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from preprocessor import Preprocessor


def main(args):

    with open(args.embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    preprocessor = Preprocessor(embeddings)

    train = preprocessor.get_dataset(args.train_path, args.n_workers)
    with open(args.output_train_path, 'wb') as f:
        pickle.dump(train, f)

    valid = preprocessor.get_dataset(args.valid_path, args.n_workers)
    with open(args.output_valid_path, 'wb') as f:
        pickle.dump(valid, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('embeddings_path', type=str,
                        help='[input] Path to the embedding pickle built with'
                             ' build_embeddings.py.')
    parser.add_argument('train_path', type=str,
                        help='[input] Path to the training json data.')
    parser.add_argument('valid_path', type=str,
                        help='[input] Path to the dev json data.')
    parser.add_argument('output_train_path', type=str,
                        help='[output] Path to the training pickle file.')
    parser.add_argument('output_valid_path', type=str,
                        help='[output] Path to the valid pickle file.')
    parser.add_argument('--n_workers', type=int, default=16)
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
