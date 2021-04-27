import torch
import torch.nn as nn

from models.commons.initializer import init_linear_wt


class LinearClassifier(nn.Module):
    """
    input   BxH tensor
    output  BxC logits
    """

    def __init__(self, encoder, enc_type, H, C):
        super(LinearClassifier, self).__init__()

        self.encoder = encoder
        self.enc_type = enc_type
        self.H = H
        self.C = C

        self.out = nn.Linear(self.H, self.C)

    def forward(self, dialogs, lens):
        if self.enc_type == 'rnn':
            _, h = self.encoder(dialogs, lens)

        return self.out(h)

