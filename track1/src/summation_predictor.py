import logging
import json
import pickle
import os
import torch
from base_predictor import BasePredictor
from modules.common import NLLLoss, RankLoss


class SummationPredictor(BasePredictor):
    def __init__(self, embeddings, hr_model_paths, rt_model_paths,
                 loss='NLLLoss', margin=0.2,
                 has_info=False, threshold=0.0, **kwargs):
        super(SummationPredictor, self).__init__(**kwargs)
        self.embeddings = torch.nn.Embedding(embeddings.size(0),
                                             embeddings.size(1))
        self.embeddings.weight = torch.nn.Parameter(embeddings)
        self.has_info = has_info

        self.n_hr_models = len(hr_model_paths)
        self.n_rt_models = len(rt_model_paths)
        models = []
        for model_path in hr_model_paths + rt_model_paths:
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path) as f:
                config = json.load(f)

            with open(config['model_parameters']['embeddings'], 'rb') as f:
                embeddings = pickle.load(f)
                config['model_parameters']['embeddings'] = \
                    embeddings.embeddings

            if config['arch'] == 'DualRNN':
                from dualrnn_predictor import DualRNNPredictor
                PredictorClass = DualRNNPredictor
            elif config['arch'] == 'HierRNN':
                from hierrnn_predictor import HierRNNPredictor
                PredictorClass = HierRNNPredictor
            elif config['arch'] in ['UttHierRNN', 'UttBinHierRNN', 'MCAN']:
                from hierrnn_predictor import UttHierRNNPredictor
                PredictorClass = UttHierRNNPredictor
                config['model_parameters']['model_type'] = config['arch']
            elif config['arch'] == 'RecurrentTransformer':
                from recurrent_transformer_predictor import RTPredictor
                PredictorClass = RTPredictor

            predictor = PredictorClass(metrics=[],
                                       **config['model_parameters'])

            # model_path = '/tmp/model.pkl'
            logging.info('loading model from {}'.format(model_path))
            predictor.load(model_path)
            models.append(predictor.model)

        self.model = Summation(models)

        # use cuda
        self.model = self.model.to(self.device)
        self.embeddings = self.embeddings.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'NLLLoss': NLLLoss(),
            'RankLoss': RankLoss(margin, threshold)
        }[loss]

    def _run_iter(self, batch, training):
        logits = self._predict_batch(batch)

        loss = self.loss(logits, batch['labels'].to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        with torch.no_grad():
            context = self.embeddings(batch['context'].to(self.device))
            options = self.embeddings(batch['options'].to(self.device))

            if self.has_info:
                prior = batch['prior'].unsqueeze(-1).to(self.device)
                suggested = batch['suggested'].unsqueeze(-1).to(self.device)
                context = torch.cat([context] + [prior, suggested] * 1, -1)

                prior = batch['option_prior'].unsqueeze(-1).to(self.device)
                suggested = batch['option_suggested'].unsqueeze(-1).to(self.device)
                options = torch.cat([options] + [prior, suggested] * 1, -1)

        logits = 0
        for i in range(self.n_hr_models):
            logits += self.model.models[i].forward(
                context.to(self.device),
                batch['utterance_ends'],
                options.to(self.device),
                batch['option_lens'])

        with torch.no_grad():
            context = self.embeddings(batch['context'].to(self.device))
            options = self.embeddings(batch['options'].to(self.device))
            if self.has_info:
                prior = batch['prior'].unsqueeze(-1).to(self.device)
                suggested = batch['suggested'].unsqueeze(-1).to(self.device)
                context = torch.cat([context] + [prior, suggested] * 3, -1)

                prior = batch['option_prior'].unsqueeze(-1).to(self.device)
                suggested = batch['option_suggested'].unsqueeze(-1).to(self.device)
                options = torch.cat([options] + [prior, suggested] * 3, -1)

        for i in range(self.n_hr_models, self.n_hr_models + self.n_rt_models):
            logits += self.model.models[i].forward(
                context.to(self.device),
                batch['utterance_ends'],
                options.to(self.device),
                batch['option_lens'])

        return logits / (self.n_hr_models + self.n_rt_models)


class Summation(torch.nn.Module):
    """

    Args:

    """
    def __init__(self, models):
        super(Summation, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, *args):
        logits = self.models[0](*args)
        for model in self.models:
            logits += model(*args)

        return logits / len(self.models)
