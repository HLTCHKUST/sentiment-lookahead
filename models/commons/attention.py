import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.commons.initializer import init_rnn_wt, init_linear_wt


class Attention(nn.Module):
    def __init__(self, H, method='general'):
        super(Attention, self).__init__()
        
        self.method = method
        
        if self.method == 'general':
            self.W = nn.Linear(H, H)
            init_linear_wt(self.W)

        elif self.method == 'concat':
            self.W = nn.Linear(H * 2, H)
            self.v = nn.Parameter(torch.FloatTensor(1, H))
            init_linear_wt(self.W)
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)
        
        self.W_c = nn.Linear(H * 2, H)
        init_linear_wt(self.W_c)

    def forward(self, K, V, Q):
        # K: Keys   -> [B, L, H]
        # V: Value  -> [B, L, H]
        # Q: Query  -> [B, 1, H]
        # ======================
        # E: Energy -> [B, 1, L]
        # returns [B, 1, H] tensor of V weighted by E

        # Calculate attention energies for each encoder output
        # and Normalize energies to weights in range 0 to 1
        e = F.softmax(self.score(K, Q), dim=2) # [B, 1, L]
        
        # re-weight values with energy
        c = torch.bmm(e, V) # [B, 1, H]
        
        h = torch.tanh(self.W_c(torch.cat((c, Q), dim=2))) # [B, 1, H]
        
        return h, e
    
    def score(self, K, Q):
        if self.method == 'dot':
            # bmm btw [B, L, H] and [B, H, 1] => [B, 1, L]
            return torch.bmm(Q, K.transpose(1, 2))
        elif self.method == 'general':
            return torch.bmm(self.W(Q), K.transpose(1, 2))
        elif self.method == 'concat': # luong attention
            B, L, _ = K.shape
            E = self.W(torch.cat((K, Q.repeat(1, L, 1)), dim=2)) # [B, L, 2H] -> [B, L, H]
            return torch.bmm(self.v.repeat(B, 1, 1), E.transpose(1, 2)) # [B, 1, H] x [B, H, L]
