import torch
import torch.nn as nn

from models.commons.initializer import init_linear_wt
from utils import constant

class BinaryClassifier(nn.Module):
    """
    input   BxH tensor
    output  BxC logits
    """

    def __init__(self, encoder, enc_type, H):
        super(BinaryClassifier, self).__init__()

        self.encoder = encoder
        self.enc_type = enc_type
        self.H = H
        self.dropout = nn.Dropout(constant.dropout)

        self.out = nn.Linear(self.H, 1)

    def forward(self, dialogs, lens=None):
        if self.enc_type == 'rnn':
            _, h = self.encoder(dialogs, lens)
        elif self.enc_type == 'bert':
            input_ids, segment_ids, input_masks = dialogs
            _, h = self.encoder(input_ids, segment_ids, input_masks, output_all_encoded_layers=False)
            h = self.dropout(h)

        return self.out(h)

    def predict_prob(self, dialogs, lens=None):
        self.eval()
        sigmoid = nn.Sigmoid()
        with torch.no_grad():
            if self.enc_type == 'rnn':
                _, h = self.encoder(dialogs, lens)
            elif self.enc_type == 'bert':
                input_ids, segment_ids, input_masks = dialogs
                _, h = self.encoder(input_ids, segment_ids, input_masks, output_all_encoded_layers=False)
        return sigmoid(self.out(h))