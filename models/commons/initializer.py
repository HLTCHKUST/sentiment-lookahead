import torch.nn as nn

from utils import constant


def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                nn.init.xavier_uniform_(wt)
                # wt.data.uniform_(-constant.rand_unif_init_mag, constant.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    # linear.weight.data.normal_(std=constant.trunc_norm_init_std)
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        n = linear.bias.size(0)
        start, end = n // 4, n // 2
        linear.bias.data.fill_(0.)
        linear.bias.data[start:end].fill_(1.)
        # linear.bias.data.nomral_(std=constant.trunc_norm_init_std)
