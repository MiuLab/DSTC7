import numpy as np
import torch


"""
A simple implementation of beam search in PyTorch. This module process one
sequence at a time.
For every decode step, you should pass the scores to searcher, and it'll return
the logits and beam indices for next step.

You can use the helper to pick next step inputs.

beam_size: the size of beams during search
max_length: the max decoding step will be executed
EOS: end of sentence logit
"""


class Searcher:
    def __init__(self, beam_size, max_length, EOS, cuda=True):
        self.beam_size = beam_size
        self.max_length = max_length
        self.EOS = EOS
        self.tt = torch.cuda if cuda else torch

    def init(self):
        self.live_beam = self.beam_size
        self.beams = np.full((self.beam_size, self.max_length), -1)
        self.pointers = np.zeros((self.beam_size, self.max_length))
        self.beam_anchors = [idx for idx in range(self.beam_size)]
        self.beam_scores = self.tt.FloatTensor(np.zeros((1, self.beam_size)))
        self.sequences = []
        self.end = False
        self.idx = 0

    def step(self, scores):
        """
        input:
            scores: [1 * dim] (first step) or [live_beam * dim] (float tensor)
        output:
            best logits:    [live_beam * 1] (long tensor)
            beam indices:   [live_beam] (long tensor)
        """
        if self.end:
            return None, None

        candidate_scores, candidate_logits = scores.topk(
            k=self.live_beam, dim=1)
        candidate_scores = (
            candidate_scores + self.beam_scores[:, :self.live_beam]).view(-1)
        candidate_logits = candidate_logits.view(-1)
        best_scores, best_indices = candidate_scores.topk(self.live_beam)
        best_logits = candidate_logits.gather(0, best_indices)
        self.beams[:self.live_beam, self.idx] = best_logits
        self.beam_scores[0, :self.live_beam] = best_scores
        best_indices = best_indices / self.live_beam
        self.pointers[:self.live_beam, self.idx] = \
            [self.beam_anchors[idx] for idx in best_indices]
        return_logits, return_indices = [], []
        self.beam_anchors = []
        if self.beams[0, self.idx] == self.EOS:
            self.end = True
            self.live_beam = 0
            self.sequences.append(self.backtrack(0, self.idx))
        else:
            for bidx, token in enumerate(
                    self.beams[:self.live_beam, self.idx]):
                if token == self.EOS:
                    self.live_beam -= 1
                    self.sequences.append(self.backtrack(bidx, self.idx))
                else:
                    self.beam_anchors.append(bidx)
                    return_logits.append(self.beams[bidx, self.idx])
                    return_indices.append(best_indices[bidx])
            if self.live_beam == 0:
                self.end = True
        self.idx += 1
        if self.idx == self.max_length:
            self.end = True

        if self.end:
            for bidx in self.beam_anchors:
                self.sequences.append(
                    self.backtrack(bidx, self.idx - 1))
            self.sort()
            return None, None

        logit_tensor = self.tt.LongTensor(return_logits).view(-1, 1)
        index_tensor = self.tt.LongTensor(return_indices)
        return logit_tensor, index_tensor

    def sort(self):
        self.sequences.sort(key=lambda x: x[1])
        self.sequences = self.sequences[::-1]

    def backtrack(self, beam_idx, idx):
        sequence = []
        score = self.beam_scores[0, beam_idx]
        anchor = [beam_idx, idx]
        while anchor[1] != -1:
            sequence.append(self.beams[anchor[0], anchor[1]])
            anchor[0] = int(self.pointers[anchor[0], anchor[1]])
            anchor[1] -= 1
        sequence = sequence[::-1]
        return [sequence, score]
