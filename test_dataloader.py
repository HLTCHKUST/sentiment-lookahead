import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from models import RNNEncoder, RNNDecoder, LinearClassifier, Seq2Seq, MultiSeq2Seq
from train_emotion import train_emotion
from train_seq2seq import eval_seq2seq, train_seq2seq
from train_multitask import train_multitask
from test import test
from utils import constant, DialogDataset, make_data_loader, collate_fn


if __name__ == "__main__":
    fasttext = True if constant.embedding == 'fasttext' else False
    aug = True if constant.parse == 'augment' else False
    sld = True if constant.parse == 'sliding' else False
    train_dataset = DialogDataset(mode='train', dataset=constant.data, sld=sld, aug=aug, path=None, load_fasttext=fasttext)
    dev_dataset = DialogDataset(mode='dev', dataset=constant.data, sld=False, aug=False, path=None, load_fasttext=fasttext)
    test_dataset = DialogDataset(mode='test', dataset=constant.data, sld=False, aug=False, path=None, load_fasttext=fasttext)
    train_dataloader = make_data_loader(train_dataset, constant.USE_CUDA, constant.embeddings_cpu, constant.B, full=constant.full, sld=sld, aug=aug, pad_idx=1, shuffle=constant.shuffle)
    dev_dataloader = make_data_loader(dev_dataset, constant.USE_CUDA, constant.embeddings_cpu, constant.B, full=constant.full, sld=sld, aug=aug, pad_idx=1, shuffle=constant.shuffle)
    test_dataloader = make_data_loader(test_dataset, constant.USE_CUDA, constant.embeddings_cpu, constant.B, full=constant.full, sld=sld, aug=aug, pad_idx=1, shuffle=constant.shuffle)
 
    print()
    for dialogs, lens, targets, target_lens, emotions in train_dataloader:
        print('train')
        print("dialogs: ", dialogs.shape)
        print("lens: ", lens.shape)
        print("targets: ", targets.shape)
        print("target_lens: ", target_lens.shape)
        print("emotions: ", emotions.shape)
        break

    print()
    for dialogs, lens, targets, target_lens, emotions in dev_dataloader:
        print('dev')
        print("dialogs: ", dialogs.shape)
        print("lens: ", lens.shape)
        print("targets: ", targets.shape)
        print("target_lens: ", target_lens.shape)
        print("emotions: ", emotions.shape)
        break

    print()
    for dialogs, lens, targets, target_lens, emotions in test_dataloader:
        print('test')
        print("dialogs: ", dialogs.shape)
        print("lens: ", lens.shape)
        print("targets: ", targets.shape)
        print("target_lens: ", target_lens.shape)
        print("emotions: ", emotions.shape)
        break