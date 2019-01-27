import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import FinalMetrics


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
        config['model_parameters']['embeddings'] = embeddings.embeddings

    logging.info('loading dev data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        config['model_parameters']['valid'] = pickle.load(f)
        config['model_parameters']['valid'].padding = \
            embeddings.to_index('</s>')
        config['model_parameters']['valid'].n_positive = config['valid_n_positive']
        config['model_parameters']['valid'].n_negative = config['valid_n_negative']
        config['model_parameters']['valid'].context_padded_len = config['context_padded_len']
        config['model_parameters']['valid'].option_padded_len = config['option_padded_len']
        config['model_parameters']['valid'].min_context_len = 10000

    logging.info('loading train data...')
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)
        train.n_positive = config['train_n_positive']
        train.n_negative = config['train_n_negative']
        train.context_padded_len = config['context_padded_len']
        train.padding = embeddings.to_index('</s>')
        train.option_padded_len = config['option_padded_len']
        train.min_context_len = 10000

    if config['arch'] == 'DualRNN':
        from dualrnn_predictor import DualRNNPredictor
        PredictorClass = DualRNNPredictor
    elif config['arch'] == 'HierRNN':
        from hierrnn_predictor import HierRNNPredictor
        PredictorClass = HierRNNPredictor
    elif config['arch'] in ['UttHierRNN', 'UttHierRNN', 'MCAN']:
        from hierrnn_predictor import UttHierRNNPredictor
        PredictorClass = UttHierRNNPredictor
        config['model_parameters']['model_type'] = config['arch']
    elif config['arch'] == 'RecurrentTransformer':
        from recurrent_transformer_predictor import RTPredictor
        PredictorClass = RTPredictor
    elif config['arch'] == 'Summation':
        from summation_predictor import SummationPredictor
        PredictorClass = SummationPredictor

    predictor = PredictorClass(
        metrics=[FinalMetrics(rank_na=config['rank_na'])],
        **config['model_parameters']
    )

    if args.load is not None:
        predictor.load(args.load)

    model_checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model.pkl'),
        'loss', 1, 'all'
    )
    metrics_logger = MetricsLogger(
        os.path.join(args.model_dir, 'log.json')
    )

    logging.info('start training!')
    predictor.fit_dataset(train,
                          train.collate_fn,
                          [model_checkpoint, metrics_logger])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str)
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
