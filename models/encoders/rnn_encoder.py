import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

from models.commons.initializer import init_rnn_wt
from utils import constant


class RNNEncoder(nn.Module):
    def __init__(self, V, D, H, L=1, embedding=None):
        super(RNNEncoder, self).__init__()
        self.V = V
        self.H = H
        self.L = L
        self.D = D
        self.bi = True if constant.bi == 'bi' else False
        self.use_lstm = constant.lstm
        # self.dropout = nn.Dropout(constant.dropout)
        
        self.cuda = constant.USE_CUDA

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(V, D)
            self.embedding.weight.requires_grad = True
        
        self.embedding_dropout = nn.Dropout(constant.dropout)

        if constant.lstm:
            self.rnn = nn.LSTM(D, H, L, batch_first=True, bidirectional=self.bi)
        else:
            self.rnn = nn.GRU(D, H, L, batch_first=True, bidirectional=self.bi)

    def soft_embed(self, x):
        # x: (T, B, V), (B, V) or (V)
        return (x.unsqueeze(len(x.shape)) * self.embedding.weight).sum(dim=len(x.shape)-1)

    def forward(self, seqs, lens, soft_encode=False, logits=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # x, lens = pad_packed_sequence(pack_sequence(seqs))
        if not soft_encode:
            x = self.embedding(seqs)
            x = self.embedding_dropout(x)
        else:
            x = self.soft_embed(logits).transpose(0, 1).contiguous()
        x = pack_padded_sequence(x, lens, batch_first=True)
        outputs, hidden = self.rnn(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        if self.use_lstm:
            h, c = hidden

        if self.bi:
            # [2, B, H] => [B, 2H]
            if self.use_lstm:
                h = h.transpose(0, 1).contiguous().view(-1, 2*self.H)
                c = c.transpose(0, 1).contiguous().view(-1, 2*self.H)
                # h = torch.cat((h[0], h[1]), 1)
                # c = torch.cat((c[0], c[1]), 1)
                return outputs, h.squeeze(), c.squeeze()
            else:
                h = torch.cat((hidden[0], hidden[1]), 1)
                return outputs, h.squeeze()
        else:
            return outputs, hidden.squeeze()

    def predict_one(self, seq):
        with torch.no_grad():
            x = self.embedding(seq)
            outputs, hidden = self.rnn(x)
            if self.bi:
                # [2, B, H] => [B, 2H]
                hidden = torch.cat((hidden[0], hidden[1]), 1)
                return outputs, hidden
            else:
                return outputs, hidden