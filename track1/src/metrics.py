import math
import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Accuracy(Metrics):
    def __init__(self, ats=[1, 5, 10, 50]):
        self.ats = ats
        self.n = 0
        self.n_correct_ats = [0] * len(self.ats)
        self.name = 'Accuracy@' + ', '.join(map(str, ats))
        self.noise = 1e-6

    def reset(self):
        self.n = 0
        self.n_correct_ats = [0] * len(self.ats)

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        # add noise to deal with cases where all predict score are identical
        predicts *= (1 + torch.rand_like(predicts) * self.noise)
        self.n += predicts.size(0)
        sorted_predicts = torch.sort(predicts)[0]
        for i, at in enumerate(self.ats):
            if predicts.size(1) <= at:
                break

            score_at = sorted_predicts[:, -at]

            # assume that the 0th option is answer
            self.n_correct_ats[i] += (predicts[:, 0] >= score_at).sum().item()

    def get_score(self):
        return [n_correct / self.n for n_correct in self.n_correct_ats]


class F1(Metrics):
    """ Precision, recall and F1 metrics.

    Args:
        threshold (float): Prediction greater than the threshold will be
            treated as positive.
    """
    name = 'Precision, Recall, F1'

    def __init__(self, threshold=0., max_selected=None):
        # true positive
        self.tp = 0

        # false positive, false negative
        self.fp, self.fn = 0, 0

        self.threshold = threshold if threshold is not None else - math.inf
        self.max_selected = max_selected
        self.noise = 1e-6

    def reset(self):
        self.tp = 0
        self.fp, self.fn = 0, 0

    def update(self, predicts, batch):
        predicts *= (1 + torch.rand_like(predicts) * self.noise)
        predicts = predicts.cpu()
        labels = batch['labels'].byte()
        assert predicts.shape == labels.shape

        # predicted positive
        pp = predicts.float() > self.threshold

        # filter not in topk
        if self.max_selected is not None:
            topk_mask = \
                predicts >= predicts.topk(self.max_selected, -1)[0][:, -1:]
            pp *= topk_mask

        # predicted negative
        pn = 1 - pp

        self.tp += (pp * labels).sum().item()
        self.fp += (pp * (1 - labels)).sum().item()
        self.fn += (pn * labels).sum().item()

    def get_score(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return [precision, recall, f1]


class MRR(Metrics):
    """
    Args:
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, rank_na=False):
        self.n = 0
        self.rank_sum = 0
        self.name = 'MRR'
        self.noise = 1e-6
        self.rank_na = rank_na

    def reset(self):
        self.n = 0
        self.rank_sum = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        if self.rank_na:
            labels = torch.cat([
                batch['labels'],
                (batch['labels'].sum(-1, keepdim=True) == 0).long()], 1
            ).float()
            predicts = torch.cat(
                [predicts, torch.zeros_like(predicts[:, :1])], 1
            )
        else:
            labels = batch['labels'].float()

        # add noise to deal with cases where all predict score are identical
        predicts *= (1 + torch.rand_like(predicts) * self.noise)
        predict_ranks = _get_rank(predicts)
        mask = labels.masked_fill(labels == 0, math.inf)
        self.rank_sum += (1 / (mask * (predict_ranks + 1)).min(-1)[0]).sum().item()
        self.n += predicts.shape[0]

    def get_score(self):
        return self.rank_sum / self.n


class Recall(Metrics):
    """
    Args:
         ats (list): @s to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, ats=[1, 5, 10, 50], rank_na=False):
        self.ats = ats
        self.n = 0
        self.n_correct_ats = [0] * len(self.ats)
        self.name = 'Recall@' + ', '.join(map(str, ats))
        self.noise = 1e-6
        self.rank_na = rank_na

    def reset(self):
        self.n = 0
        self.n_correct_ats = [0] * len(self.ats)

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        if self.rank_na:
            labels = torch.cat([
                batch['labels'],
                (batch['labels'].sum(-1, keepdim=True) == 0).long()], 1
            ).float()
            predicts = torch.cat([predicts,
                                  torch.zeros_like(predicts[:, :1])], 1)
        else:
            labels = batch['labels'].float()

        # add noise to deal with cases where all predict score are identical
        predicts *= (1 + torch.rand_like(predicts) * self.noise)
        self.n += predicts.shape[0]
        predict_ranks = _get_rank(predicts)
        mask = labels.masked_fill(labels == 0, math.inf)
        mask_predict_ranks = mask * predict_ranks
        for i, at in enumerate(self.ats):
            recall = ((mask_predict_ranks < at).float().sum(-1) /
                      labels.float().sum(-1)).sum()
            self.n_correct_ats[i] += recall.item()

    def get_score(self):
        return [n_correct / self.n for n_correct in self.n_correct_ats]


class FinalMetrics(Metrics):
    """
    Args:
         ats (list): @s to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, ats=[1, 5, 10, 50], rank_na=False):
        self.at10ind = ats.index(10)
        self.recall = Recall(ats, rank_na)
        self.mrr = MRR(rank_na)
        self.name = '{}, {}, avg.'.format(self.recall.name, self.mrr.name)
        self.noise = 1e-6

    def reset(self):
        self.recall.reset()
        self.mrr.reset()

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        self.recall.update(predicts, batch)
        self.mrr.update(predicts, batch)

    def get_score(self):
        return [
            self.recall.get_score(), self.mrr.get_score(),
            (self.recall.get_score()[self.at10ind] + self.mrr.get_score()) / 2
        ]


def _get_rank(tensor):
    _, indices = (-tensor).sort(-1)
    rank = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            rank[i, indices[i, j]] = j

    return rank
