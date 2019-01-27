import csv
import random
import shutil
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import ruamel_yaml
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from sumeval.metrics.rouge import RougeCalculator
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from tqdm import tqdm
from visdom import Visdom

from constants import SEED, DEVICE
from utils import onehot


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings(
    module='nltk', category=UserWarning,
    message='\nThe hypothesis contains 0 counts of \d-gram overlaps\.', action='ignore')


class Trainer:
    def __init__(self, config_path, train_cfg, vocab, train_data_loader,
                 dev_data_loader, model):
        self.train_cfg = train_cfg
        self.vocab = vocab
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.model = model

        self.exp_dir = Path(train_cfg.exp_dir, train_cfg.exp_name)
        try:
            (self.exp_dir / 'ckpts').mkdir(parents=True)
            (self.exp_dir / 'samples').mkdir(parents=True)
            print(f'[-] Experiment directory created at {self.exp_dir}')
        except FileExistsError:
            print(f'[!] Directory {self.exp_dir} already exists')
            exit(1)
        shutil.copy(config_path, self.exp_dir)

        self.init_stats_and_plots()
        self.rouge = RougeCalculator(stopwords=False)
        self.meteor = Meteor()

    def init_stats_and_plots(self):
        self.iter_count = 0
        self.metrics = [
            'decode_loss', 'kld_loss_ratio', 'kld_loss', 'loss', 'rouge_l', 'bleu_1',
            'bleu_2', 'bleu_3', 'meteor']

        fieldnames = [f'{mode}_{metric}' for mode, metric
                      in product(['train', 'eval'], self.metrics)]
        self.stats_writer = csv.DictWriter(
            (self.exp_dir / 'stats.csv').open(mode='w', buffering=1),
            fieldnames=fieldnames)
        self.stats_writer.writeheader()

        self.vis = Visdom(port=self.train_cfg.visdom_port, env=self.train_cfg.exp_name)
        self.vis.close()
        for m in self.metrics:
            self.vis.line(win=m, X=np.array([0]), Y=np.array([[0, 0]]), opts={
                'title': m,
                'xlabel': 'epoch',
                'legend': ['train', 'eval']
            })
        self.first_plot = True

        self.reset_stats_and_samples()

    def reset_stats_and_samples(self):
        self.stats = {
            'train': {k: [] for k in self.metrics},
            'eval': {k: [] for k in self.metrics}
        }
        self.samples = {'train': [], 'eval': []}

    def start(self):
        tqdm.write('[-] Start training!')
        bar = tqdm(
            range(1, self.train_cfg.n_epochs + 1), desc='[Total progress]', leave=False,
            position=0)
        for epoch in bar:
            self.epoch = epoch
            self.run_epoch('train')
            self.run_epoch('eval')
            self.write_stats()
            self.write_samples()
            self.plot()
            self.save_ckpt()
            self.reset_stats_and_samples()
        tqdm.write('[-] Training done!')
        bar.close()

    def run_epoch(self, mode):
        if mode == 'train':
            data_loader = self.train_data_loader
            self.model.set_train()
            desc_prefix = 'Train'
        elif mode == 'eval':
            data_loader = self.dev_data_loader
            self.model.set_eval()
            desc_prefix = 'Eval '
        torch.set_grad_enabled(mode == 'train')

        bar = tqdm(
            data_loader, desc=f'[{desc_prefix} epoch {self.epoch:2}]', leave=False,
            position=1)
        metric_bar = tqdm(
            [0], desc='[Metric display]', bar_format='{desc}: {postfix}', leave=False,
            position=2)
        for batch_idx, batch in enumerate(bar):
            if mode == 'train':
                self.iter_count += 1

            context = batch['context'].to(device=DEVICE)
            fact = batch['fact'].to(device=DEVICE)
            response = batch['response'].to(device=DEVICE)
            context_pad_mask = batch['context_pad_mask'].to(device=DEVICE)
            fact_pad_mask = batch['fact_pad_mask'].to(device=DEVICE)
            response_pad_mask = batch['response_pad_mask'].to(device=DEVICE)

            if mode == 'train':
                logits, latent_mean, latent_log_var = self.model(
                    context, fact, response, context_pad_mask, fact_pad_mask,
                    response_pad_mask)
                predictions = logits.max(dim=-1)[1].cpu()

                batch_size, _, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                response_flat = response.reshape(-1)
                decode_loss = F.cross_entropy(
                    logits_flat, response_flat, ignore_index=self.vocab.word.sp.pad.idx)
                self.stats[mode]['decode_loss'].append(decode_loss.item())

                if latent_mean is not None and latent_log_var is not None:
                    kld_loss = -0.5 * (1 + latent_log_var - latent_mean ** 2 - latent_log_var.exp()).sum()
                    self.stats[mode]['kld_loss'].append(kld_loss.item())
                    kld_loss_ratio = self.train_cfg.kld_loss_init_ratio * \
                        self.train_cfg.kld_loss_anneal_rate ** self.iter_count
                    kld_loss_ratio = min(kld_loss_ratio, 1)
                    self.stats[mode]['kld_loss_ratio'] = [kld_loss_ratio]
                    loss = decode_loss + kld_loss_ratio * kld_loss
                    self.stats[mode]['loss'].append(loss.item())
                else:
                    self.stats[mode]['kld_loss'].append(0)
                    self.stats[mode]['kld_loss_ratio'] = [0]
                    loss = decode_loss
                    self.stats[mode]['loss'].append(loss.item())

                self.model.zero_grad()
                loss.backward()
                if self.train_cfg.max_grad_norm > 0:
                    self.model.clip_grad(self.train_cfg.max_grad_norm)
                self.model.update()
            elif mode == 'eval':
                predictions, latent_mean, latent_log_var = self.model.infer(
                    context, fact, context_pad_mask, fact_pad_mask,
                    beam_size=self.train_cfg.beam_size)
                self.stats[mode]['decode_loss'].append(0)
                self.stats[mode]['kld_loss'].append(0)
                self.stats[mode]['kld_loss_ratio'] = [0]
                self.stats[mode]['loss'].append(0)

            for i in range(len(predictions)):
                hyp = [w for w in predictions[i] if w not in
                       [self.vocab.word.sp.unk.idx, self.vocab.word.sp.pad.idx]]
                try:
                    eos_idx = hyp.index(self.vocab.word.sp.eos.idx)
                except ValueError:
                    pass
                else:
                    hyp = hyp[:eos_idx]
                hyp = [self.vocab.word.itov(i) for i in hyp]
                ref = [self.vocab.full.itov(i) for i in batch['orig_response'][i]]

                rouge_l = self.rouge.rouge_l(hyp, [ref]) * 100
                bleu_1 = sentence_bleu([ref], hyp, [1]) * 100
                bleu_2 = sentence_bleu([ref], hyp, [1 / 2] * 2) * 100
                bleu_3 = sentence_bleu([ref], hyp, [1 / 3] * 3) * 100
                meteor = self.meteor.compute_score(
                    {0: [' '.join(ref)]}, {0: [' '.join(hyp)]})[0] * 100

                self.stats[mode]['rouge_l'].append(rouge_l)
                self.stats[mode]['bleu_1'].append(bleu_1)
                self.stats[mode]['bleu_2'].append(bleu_2)
                self.stats[mode]['bleu_3'].append(bleu_3)
                self.stats[mode]['meteor'].append(meteor)

                if batch_idx == len(data_loader) - 1:
                    context = [self.vocab.word.itov(i) for i in batch['context'][i]]
                    fact = [self.vocab.word.itov(i) for i in batch['fact'][i]]
                    response_ref_model = \
                        [self.vocab.word.itov(i) for i in batch['response'][i]]
                    response_hyp_model = \
                        [self.vocab.word.itov(i) for i in predictions[i]]
                    self.samples[mode].append({
                        'context': ' '.join(context),
                        'fact': ' '.join(fact),
                        'response_ref': ' '.join(ref),
                        'response_hyp': ' '.join(hyp),
                        'response_ref_model': ' '.join(response_ref_model),
                        'response_hyp_model': ' '.join(response_hyp_model),
                        'rouge_l': rouge_l,
                        'bleu_1': bleu_1,
                        'bleu_2': bleu_2,
                        'bleu_3': bleu_3,
                        'meteor': meteor,
                    })

            metric_bar.set_postfix_str(
                '\b\b' + ', '.join([f'{m}: {np.mean(self.stats[mode][m]):5.2f}'
                                    for m in self.metrics]))

        for metric in self.metrics:
            self.stats[mode][metric] = np.mean(self.stats[mode][metric])

        bar.close()
        metric_bar.close()

    def write_stats(self):
        self.stats_writer.writerow(
            {f'{mode}_{metric}': self.stats[mode][metric]
             for mode, metric in product(['train', 'eval'], self.metrics)})

    def write_samples(self):
        sample_path = self.exp_dir / 'samples' / f'epoch-{self.epoch}.yaml'
        with sample_path.open(mode='w') as f:
            ruamel_yaml.dump(self.samples, f, default_flow_style=False)

    def plot(self):
        if self.first_plot:
            update_mode = 'replace'
            self.first_plot = False
        else:
            update_mode = 'append'

        for m in self.metrics:
            self.vis.line(
                win=m, update=update_mode, X=np.array([self.epoch]),
                Y=np.array([[self.stats['train'][m], self.stats['eval'][m]]]))

        self.vis.save([self.train_cfg.exp_name])

    def save_ckpt(self):
        self.model.save_state(self.epoch, self.stats, self.exp_dir / 'ckpts')
