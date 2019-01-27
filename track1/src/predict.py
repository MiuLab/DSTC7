import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
import torch

from metrics import FinalMetrics


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
        config['model_parameters']['embeddings'] = embeddings.embeddings

    if 'valid' in config['model_parameters']:
        logging.info('loading dev data...')
        with open(config['model_parameters']['valid'], 'rb') as f:
            config['model_parameters']['valid'] = pickle.load(f)
            config['model_parameters']['valid'].padding = \
                embeddings.to_index('</s>')
            config['model_parameters']['valid'].n_positive = \
                config['valid_n_positive']
            config['model_parameters']['valid'].n_negative = \
                config['valid_n_negative']
            config['model_parameters']['valid'].context_padded_len = \
                config['context_padded_len']
            config['model_parameters']['valid'].option_padded_len = \
                config['option_padded_len']
            config['model_parameters']['valid'].min_context_len = 10000

    logging.info('loading test data...')
    with open(config['test'], 'rb') as f:
        test = pickle.load(f)
        test.padding = embeddings.to_index('</s>')
        test.n_positive = 0
        test.n_negative = 100
        test.context_padded_len = config['context_padded_len']
        test.option_padded_len = config['option_padded_len']
        test.min_context_len = 10000

    if config['arch'] == 'DualRNN':
        from dualrnn_predictor import DualRNNPredictor
        PredictorClass = DualRNNPredictor
    elif config['arch'] == 'HierRNN':
        from hierrnn_predictor import HierRNNPredictor
        PredictorClass = HierRNNPredictor
    elif config['arch'] == 'UttHierRNN' or config['arch'] == 'UttBinHierRNN':
        from hierrnn_predictor import UttHierRNNPredictor
        PredictorClass = UttHierRNNPredictor
        config['model_parameters']['model_type'] = config['arch']
    elif config['arch'] == 'RecurrentTransformer':
        from recurrent_transformer_predictor import RTPredictor
        PredictorClass = RTPredictor
    elif config['arch'] == 'Summation':
        from summation_predictor import SummationPredictor
        PredictorClass = SummationPredictor

    predictor = PredictorClass(metrics=[],
                               **config['model_parameters'])

    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))

    # model_path = '/tmp/model.pkl'
    if not args.not_load:
        logging.info('loading model from {}'.format(model_path))
        predictor.load(model_path)

    if 'valid' in config['model_parameters']:
        logging.info('predicting valid set...')
        predicts = predictor.predict_dataset(
            config['model_parameters']['valid'],
            config['model_parameters']['valid'].collate_fn)

        labels = torch.tensor(
            [sample['labels']
             for sample in config['model_parameters']['valid']]
        )
        final = FinalMetrics(rank_na=config['rank_na'])
        final.update(predicts, {'labels': labels})
        print(final.get_score())
        output_path = os.path.join(args.model_dir,
                                   'predict-valid-{}.json'.format(args.epoch))
        write_predict(predicts, config['model_parameters']['valid'],
                      output_path, config['rank_na'])

    logging.info('predicting...')
    predicts = predictor.predict_dataset(test, test.collate_fn)

    output_path = os.path.join(args.model_dir,
                               'predict-{}.json'.format(args.epoch))
    write_predict(predicts, test, output_path, config['rank_na'])


def write_predict(predicts, data, output_path, rank_na=False):
    outputs = []
    for predict, sample in zip(predicts, data):
        candidate_ranking = [
            {
                'candidate-id': oid,
                'confidence': score.item()
            }
            for score, oid in zip(predict, sample['option_ids'])
        ]
        if rank_na:
            candidate_ranking.append({
                'candidate-id': "NONE",
                'confidence': 0
            })

        candidate_ranking = sorted(candidate_ranking,
                                   key=lambda x: -x['confidence'])
        outputs.append({
            'example-id': sample['id'],
            'candidate-ranking': candidate_ranking
        })

    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent='    ')


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--not_load', action='store_true',
                        help='Do not load any model.')
    parser.add_argument('--epoch', type=int, default=10)
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
