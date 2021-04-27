import sys
import os
import time

import torch
import torch.nn.functional as F

try:
    from utils import constant
except ImportError:
    import constant

"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = constant.pad_idx
        self.bos = constant.sou_idx
        self.eos = constant.eou_idx
        self.t = torch.cuda if constant.USE_CUDA else torch

        # The score for each translation on the beam.
        self.scores = self.t.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.t.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    def __str__(self):
        s = " \n \
        Beam Search Object: \n \
        Beam Size: {}\n \
        Pad IDX: {}\n \
        Start IDX: {}\n \
        End IDX: {}\n \
        Scores: {}\n \
        Prev Ks: {}\n \
        Next Ys: {}\n \
        "
        return s.format(self.size, self.pad, self.bos, self.eos, \
                        self.scores, self.prevKs, self.nextYs)

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk`
    #  : Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- + log probs of advancing from the last step (K x V)
    #                 K is what? => beam size
    # Returns: True if beam search is complete.

    def advance(self, wordLk):
        """Advance the beam."""
        num_words = wordLk.size(1)

        # # force the output to be longer than self.min_length
        # cur_len = len(self.next_ys)
        # if cur_len < self.min_length:
        #     for k in range(len(word_probs)):
        #         word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self.eos:
                    beam_lk[i] = -1e20
        else:
            beam_lk = wordLk[0]
            
        print(beam_lk)
        flat_beam_lk = beam_lk.view(-1) # squeeze

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened (K, K*V) array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        # print(bestScores)
        # print(bestScoresId)
        # print(prev_k)
        # print(prev_k * num_words)
        # print(bestScoresId - prev_k * num_words)
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words) # V+1th word => 0th word 

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        for i in range(1, self.size):
            if self.nextYs[-1][i] == self.eos:
                self.scores[i] = -1e10

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     The hypothesis
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]


if __name__ == "__main__":
    beam = Beam(constant.beam_size)
    print(beam)

    V = 5
    words = ['']
    probs = beam.t.distributions.normal.Normal(1.0, 2).sample((V,))
    # probs = beam.t.distributions.normal.Normal(1.0, 2).sample((beam.size, V))
    probs = F.log_softmax(probs, dim=0)
    probs = beam.t.Tensor([0.35, 0.3, 0.2, 0.1, 0.05])
    # print(probs.max(), probs.argmax())
    # print(probs.shape)
    # probs = probs.unsqueeze(1)
    # print(probs.shape)
    probs = probs.repeat(beam.size, 1)
    print(probs.shape)
    # probs = probs.unsqueeze(1).repeat(1, beam.size, 1)
    # print(probs.shape)
    topv, topi = probs.topk(beam.size, 1, True, True)
    print(topv)
    print(topi)
    print()
    while not beam.done:
        beam.advance(probs)
        print(beam)
        beam.advance(probs)
        print(beam)
        beam.advance(probs)
        print(beam)
        beam.advance(probs)
        print(beam)
        beam.advance(probs)
        print(beam)
        print(beam.get_hyp(0))
        print(beam.get_best())
        # beam.advance(probs)
        # print(beam)
        # beam.advance(probs)
        # print(beam)
        # beam.advance(probs)
        # print(beam)
        break