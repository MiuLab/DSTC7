import argparse
import pdb
import sys
import traceback
import json
import pickle
import logging
import torch
import spacy
from embeddings import Embeddings


def main(args):
    nlp = spacy.load('en_core_web_sm',
                     disable=['tagger', 'ner', 'parser', 'textcat'])

    with open(args.info_path) as f:
        courses = json.load(f)

    with open(args.orig_embedding_path, 'rb') as f:
        orig_embeddings = pickle.load(f)

    embeddings = Embeddings(args.embedding_path, None)
    for _, course in courses.items():
        name = course['Course']
        vec = torch.stack(
            [embeddings.embeddings[embeddings.to_index(w.text)]
             for w in nlp(course['Description'])],
            dim=0
        ).mean(0)
        if name in orig_embeddings.word_dict:
            orig_embeddings.embeddings[
                orig_embeddings.to_index(name)
            ] = vec
        else:
            orig_embeddings.add(name, vec)

    with open(args.output_path, 'wb') as f:
        pickle.dump(orig_embeddings, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Init course embeddings with course information."
    )
    parser.add_argument('info_path', type=str,
                        help='')
    parser.add_argument('orig_embedding_path', type=str,
                        help='')
    parser.add_argument('embedding_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
