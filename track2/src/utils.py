import mmap

import torch


def count_lines(file_path):
    with file_path.open(mode='r+') as f:
        buf = mmap.mmap(f.fileno(), 0)
        cnt = 0
        while buf.readline():
            cnt += 1

    return cnt


def onehot(x, n):
    onehot = torch.zeros(*x.shape, n, device=x.device)
    onehot = onehot.scatter(x.dim(), x.unsqueeze(-1), 1)

    return onehot
