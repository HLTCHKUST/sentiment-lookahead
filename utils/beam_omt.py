from __future__ import division

import torch
import torch.nn.functional as F

try:
    from utils import constant
except ImportError:
    import constant

# Code borrowed from OpenNMT
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate


class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, 
                 n_best=1,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size
        self.tt = torch.cuda if constant.USE_CUDA else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        # .fill_(constant.pad_idx)]
                        .fill_(constant.sou_idx)]
        self.next_ys[0][0] = constant.sou_idx

        # Has EOS topped the beam yet.
        self._eos = constant.eou_idx
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def __str__(self):
        s = " \n \
        Beam Search Object: \n \
        Beam Size: {}\n \
        End IDX: {}\n \
        Scores: {}\n \
        Prev Ks: {}\n \
        Next Ys: {}\n \
        "
        return s.format(self.size, self._eos, self.scores, \
                        self.prev_ks, self.next_ys)#, self.finished)

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.
        Parameters:
        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step
        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        # if self.stepwise_penalty:
        #     self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened (K, K*V) array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words)) # V+1th word => 0th word 
        # self.attn.append(attn_out.index_select(0, prev_k))
        # self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            # attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], None #, torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`
    Args:
       alpha (float): length parameter
       beta (float): coverage parameter
       coverage_penalty (float): coverage_penalty
       length_penalty (float): length_penalty
    """

    def __init__(self, alpha=0.8, beta=5, coverage_penalty='none', length_penalty='wu'):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(coverage_penalty, length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       None,
                                       #beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"], #+ attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attentions"
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty


class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.
    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """

    def __init__(self, cov_pen, length_pen):
        self.length_pen = length_pen
        self.cov_pen = cov_pen

    def coverage_penalty(self):
        if self.cov_pen == "wu":
            return self.coverage_wu
        elif self.cov_pen == "summary":
            return self.coverage_summary
        else:
            return self.coverage_none

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    """
    Below are all the different penalty terms implemented so far
    """

    def coverage_wu(self, beam, cov, beta=0.):
        """
        NMT coverage re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """
        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        return beta * penalty

    def coverage_summary(self, beam, cov, beta=0.):
        """
        Our summary penalty.
        """
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        return beta * penalty

    def coverage_none(self, beam, cov, beta=0.):
        """
        returns zero as penalty
        """
        return beam.scores.clone().fill_(0.0)

    def length_wu(self, beam, logprobs, alpha=0.):
        """
        NMT length re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        modifier = (((5 + len(beam.next_ys)) ** alpha) /
                    ((5 + 1) ** alpha))
        return (logprobs / modifier)

    def length_average(self, beam, logprobs, alpha=0.):
        """
        Returns the average probability of tokens in a sequence.
        """
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0., beta=0.):
        """
        Returns unmodified scores.
        """
        return logprobs


if __name__ == "__main__":
    beam = Beam(constant.beam_size, 
                global_scorer=GNMTGlobalScorer(), 
                cuda=constant.USE_CUDA)
    print(beam)

    V = 5
    words = ['']
    probs = beam.tt.distributions.normal.Normal(1.0, 2).sample((V,))
    # probs = beam.t.distributions.normal.Normal(1.0, 2).sample((beam.size, V))
    probs = F.log_softmax(probs, dim=0)
    probs = beam.tt.Tensor([0.35, 0.3, 0.2, 0.1, 0.05])
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

    beam.advance(probs)
    print(beam)
    beam.advance(probs)
    print(beam)
    beam.advance(probs)
    print(beam)
    # beam.advance(probs)
    # print(beam)
    # beam.advance(probs)
    # print(beam)