import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.commons.attention import Attention
from models.commons.initializer import init_rnn_wt, init_linear_wt
from utils import constant


class RNNDecoder(nn.Module):
    def __init__(self, V, D, H, L=1, embedding=None):
        super(RNNDecoder, self).__init__()
        self.V = V
        self.H = H
        self.L = L
        self.D = D
        if constant.attn != 'none':
            self.attention = Attention(H, constant.attn)
        # self.dropout = nn.Dropout(constant.dropout)
        
        self.cuda = constant.USE_CUDA
        self.embeddings_cpu = constant.embeddings_cpu

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(V, D)
            self.embedding.weight.requires_grad = True

        if constant.lstm:
            self.rnn = nn.LSTM(D, H, L, batch_first=True, bidirectional=False)
        else:
            self.rnn = nn.GRU(D, H, L, batch_first=True, bidirectional=False)
            
        self.out = nn.Linear(H, V)
        if constant.weight_tie:
            self.out = nn.Linear(H, V)
            self.out.weight = self.embedding.weight # Assuming H == D. They share the weight, and updated together

    def forward(self, x_t, last_h, src_hs=None, use_attn=False):
        # Note: we run this in a for loop (mulitple batches over single token at a time)
        # batch_size = x_t.size(0)
        x = self.embedding(x_t)
        if self.cuda and self.embeddings_cpu:
            x = x.cuda()
        # x = self.dropout(x)
        # x = x.view(1, batch_size, self.H) # S=1 x B x N
        outputs, dec_h_t = self.rnn(x.unsqueeze(1), last_h) # [B, 1, H] & [1, B, H]
        
        if use_attn:
            h, _ = self.attention(src_hs, src_hs, outputs)
            # output = self.out(self.linear(h))
            output = self.out(h)
        else:
            # output = self.out(self.linear(outputs))
            output = self.out(outputs)

        return output.squeeze(), dec_h_t

    def predict_one(self, x_t, last_h, src_hs=None, use_attn=False):
        with torch.no_grad():
            x = self.embedding(x_t)
            outputs, dec_h_t = self.rnn(x.unsqueeze(1), last_h) # [B, 1, H] & [1, B, H]
            if use_attn:
                h, _ = self.attention(src_hs, src_hs, outputs)
                output = self.out(h)
            else:
                output = self.out(outputs)
            return output.squeeze(), dec_h_t