import random
from collections import deque

import dill as pickle
import numpy as np
from tqdm import tqdm
import spacy

import torch
import torch.nn as nn

from models import BinaryClassifier, RNNEncoder, RNNDecoder, RLSeq
from test import test
from utils import constant
from utils.utils import load_model


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, context):
        self.vocab = vocab
        self.context = self.vocab.transform_one(context)
        self.context_len = len(self.context)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # here we ignore index since we only have one input
        return self.context, self.context_len

def collate_fn(batch):
    context, context_len = zip(*batch)
    context = torch.from_numpy(np.array(context))
    if constant.USE_CUDA:
        context = context.cuda()
    return context, context_len

def batchify(lang, context):
    if type(context) != 'str':
        c = ''
        for ctx in context:
            c += ctx + ' '
    return iter(torch.utils.data.DataLoader(dataset=SingleDataset(lang, c), batch_size=1, shuffle=False, collate_fn=collate_fn)).next()


if __name__ == "__main__":
    CONTEXT_SIZE = 3
    C = constant.C
    H = constant.H
    D = constant.D

    with open('data/prep/empathetic-dialogue/lang_shared.pkl', 'rb') as f:
        lang = pickle.load(f)
    V = len(lang)

    # define and load policy model
    encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=None)
    decoder = RNNDecoder(V=V, D=D, H=H, L=1, embedding=None)
    model = RLSeq(encoder=encoder, decoder=decoder, vocab=lang)

    constant.bi = 'none'
    reward_model = BinaryClassifier(encoder=RNNEncoder(V=V, D=D, H=300, L=1), enc_type='rnn', H=300)
    constant.bi = 'bi'
    model.init_reward(reward_model)
    model.init_baseline_reward()

    model = load_model(model, constant.test_path)
    model.eval()
    # context = 'hello my name is Midnight'
    # x, _ = batchify(lang, context)
    # sent = model.predict_one(x)
    # sent = " ".join([lang.index2word[x_t] for x_t in iter(lambda x=iter(sent.data.numpy()): next(x), constant.eou_idx)])

    print()
    print()
    print('Start to chat')
    context = deque(CONTEXT_SIZE * [''], maxlen=CONTEXT_SIZE)
    while(True):
        msg = input(">>> ")
        if msg == 'reset context':
            context = deque(CONTEXT_SIZE * [''], maxlen=CONTEXT_SIZE)
            continue
        if len(str(msg).rstrip().lstrip()) != 0:
            context.append(str(msg).rstrip().lstrip())
            x, _ = batchify(lang, context)
            sent = model.predict_one(x)
            sent = " ".join([lang.index2word[x_t] for x_t in iter(lambda x=iter(sent.data.numpy()): next(x), constant.eou_idx)])
            print(sent)
            # context.append(sent)
            print()
