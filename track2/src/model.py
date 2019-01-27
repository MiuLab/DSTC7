from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

import constants
from module import Net


torch.manual_seed(constants.SEED)
torch.cuda.manual_seed_all(constants.SEED)


class Model:
    def __init__(self, vocab, net_cfg, optim_cfg):
        print(f'[*] Creating model')
        self.__net = Net(vocab, **net_cfg)
        self.__net.to(device=constants.DEVICE)
        print(f'[-] Model created')

        optim = getattr(torch.optim, optim_cfg.algo)
        self.__optim = optim(
            filter(lambda p: p.requires_grad, self.__net.parameters()),
            **optim_cfg.kwargs)

    def set_train(self):
        self.__net.train()

    def set_eval(self):
        self.__net.eval()

    def __call__(self, *args, **kwargs):
        return self.__net(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self.__net.infer(*args, **kwargs)

    def zero_grad(self):
        self.__optim.zero_grad()

    def clip_grad(self, max_grad_norm):
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.__net.parameters()), max_grad_norm)

    def update(self):
        self.__optim.step()

    def save_state(self, epoch, stats, ckpt_dir):
        ckpt_path = ckpt_dir / f'epoch-{epoch}.ckpt'
        tqdm.write(f'[*] Saving model state')
        torch.save({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'stats': stats,
            'net_state': self.__net.state_dict(),
            'optim_state': self.__optim.state_dict()
        }, ckpt_path)
        tqdm.write(f'[-] Model state saved to {ckpt_path}')

    def load_state(self, ckpt_path):
        print(f'[*] Loading model state')
        ckpt = torch.load(ckpt_path)
        self.__net.load_state_dict(ckpt['net_state'])
        self.__net.to(constants.DEVICE)
        self.__optim.load_state_dict(ckpt['optim_state'])
        for state in self.__optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=constants.DEVICE)
        print(f'[-] Model state loaded from {ckpt_path}')
