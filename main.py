from copy import deepcopy
import random

import numpy as np
from tqdm import tqdm
import spacy

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertForSequenceClassification

from models import RNNEncoder, RNNDecoder, LinearClassifier, BinaryClassifier, Seq2Seq, RLSeq, MultiSeq2Seq, LVED
from train_emotion import train_emotion
from train_sentiment import train_sentiment
from train_topic import train_topic
from train_seq2seq import train_seq2seq
from train_rl import train_rl
from train_multitask import train_multitask
from train_lved import train_lved
from test import test
from utils import constant, DialogDataset, make_dialog_data_loader, SentimentDataset, make_sentiment_data_loader
from utils.utils import load_model


# Usage: python main.py --task emotion --data dailydialog --B 64 --full --fasttext --H 300 --D 300 --L 1
if __name__ == "__main__":
    fasttext = True if constant.embedding == 'fasttext' else False

    if constant.task != 'sentiment':
        usr = True if constant.parse == 'user' else False
        sys = True if constant.parse == 'system' else False
        train_dataset = DialogDataset(mode='train', dataset=constant.data, sys=sys, usr=usr, path=None, load_fasttext=fasttext)
        train_dataloader = make_dialog_data_loader(train_dataset, constant.USE_CUDA, constant.embeddings_cpu, constant.B, pad_idx=1, shuffle=constant.shuffle)
        if constant.eval_parse:
            usr = True if constant.parse == 'user' else False
            sys = True if constant.parse == 'system' else False
        else:
            usr = False
            sys = False
        dev_dataset = DialogDataset(mode='dev', dataset=constant.eval_data, sys=sys, usr=usr, path=None, load_fasttext=fasttext)
        test_dataset = DialogDataset(mode='test', dataset=constant.eval_data, sys=sys, usr=usr, path=None, load_fasttext=fasttext)
        dev_dataset.lang = train_dataset.lang
        test_dataset.lang = train_dataset.lang
        dev_dataset.fasttext = train_dataset.fasttext
        test_dataset.fasttext = train_dataset.fasttext
        dev_dataloader = make_dialog_data_loader(dev_dataset, constant.USE_CUDA, constant.embeddings_cpu, constant.B, pad_idx=1, shuffle=constant.shuffle)
        test_dataloader = make_dialog_data_loader(test_dataset, constant.USE_CUDA, constant.embeddings_cpu, constant.B, pad_idx=1, shuffle=constant.shuffle)
    else:
        train_dataset = SentimentDataset(mode='train', dataset=constant.data, load_fasttext=fasttext)
        train_dataloader = make_sentiment_data_loader(train_dataset, constant.USE_CUDA, constant.B, pad_idx=1, shuffle=constant.shuffle)
        dev_dataset = SentimentDataset(mode='dev', dataset=constant.eval_data, load_fasttext=fasttext)
        test_dataset = SentimentDataset(mode='test', dataset=constant.eval_data, load_fasttext=fasttext)
        dev_dataset.lang = train_dataset.lang
        test_dataset.lang = train_dataset.lang
        dev_dataset.fasttext = train_dataset.fasttext
        test_dataset.fasttext = train_dataset.fasttext
        dev_dataloader = make_sentiment_data_loader(dev_dataset, constant.USE_CUDA, constant.B, pad_idx=1, shuffle=constant.shuffle)
        test_dataloader = make_sentiment_data_loader(test_dataset, constant.USE_CUDA, constant.B, pad_idx=1, shuffle=constant.shuffle)

    dataloaders = (train_dataloader, dev_dataloader, test_dataloader)

    C = constant.C
    H = constant.H
    D = constant.D
    V = len(train_dataset.lang)

    # Shared Encoder-Decoder Embedding
    embedding = None
    if constant.share_embeddings:
        embedding = nn.Embedding(V, D)
        if constant.embedding == 'fasttext':
            embedding.weight = nn.Parameter(torch.from_numpy(train_dataset.fasttext).float())
            embedding.weight.requires_grad = constant.update_embeddings

    if constant.task == 'multiseq':
        encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
        decoder = RNNDecoder(V=V, D=D, H=H, L=1, embedding=embedding)
        if constant.share_rnn:
            decoder.rnn = encoder.rnn
        model = MultiSeq2Seq(C=C, encoder=encoder, decoder=decoder, vocab=train_dataset.lang)
        if constant.policy_model != '':
            seq2seq = load_model(Seq2Seq(encoder=encoder, decoder=decoder, vocab=train_dataset.lang), constant.policy_model)
            model.encoder = deepcopy(seq2seq.encoder)
            model.decoder = deepcopy(seq2seq.decoder)
            if constant.bi == 'bi':
                model.reduce_state = deepcopy(seq2seq.reduce_state)
            del seq2seq
        train_fn = train_multitask
    elif constant.task == 'emotion':
        encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
        if constant.bi == 'bi':
            H = H * 2
        model = LinearClassifier(encoder=encoder, enc_type='rnn', H=H, C=C)
        train_fn = train_emotion
    elif constant.task == 'sentiment':
        if constant.use_bert:
            encoder = BertModel.from_pretrained('bert-base-cased')
            # model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=1)
            model = BinaryClassifier(encoder=encoder, enc_type='bert', H=H)
        else:
            encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
            if constant.bi == 'bi':
                H = H * 2
            model = BinaryClassifier(encoder=encoder, enc_type='rnn', H=H)
        train_fn = train_sentiment
    elif constant.task == 'seq2seq':
        encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
        decoder = RNNDecoder(V=V, D=D, H=H, L=1, embedding=embedding)
        if constant.share_rnn:
            decoder.rnn = encoder.rnn
        model = Seq2Seq(encoder=encoder, decoder=decoder, vocab=train_dataset.lang)
        train_fn = train_seq2seq
    elif constant.task == 'rlseq':
        # define and load policy model
        encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
        decoder = RNNDecoder(V=V, D=D, H=H, L=1, embedding=embedding)
        model = RLSeq(encoder=encoder, decoder=decoder, vocab=train_dataset.lang)

        model = load_model(model, constant.policy_model)
        model.vocab.nlp.tokenizer.add_special_case(u"__unk__", [{spacy.symbols.ORTH: u"__unk__"}])

        # define and load user model
        if constant.use_user:
            encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
            decoder = RNNDecoder(V=V, D=D, H=H, L=1, embedding=embedding)
            user_model = Seq2Seq(encoder=encoder, decoder=decoder, vocab=train_dataset.lang)
            user_model = load_model(user_model, constant.user_model)
            model.init_user_model(user_model)

        # define and load reward model
        if constant.use_bert:
            reward_model = BinaryClassifier(encoder=BertModel.from_pretrained('bert-base-cased'), enc_type='bert', H=768)
        else:
            if constant.reward_model == 'saved/rnn_trace_model':
                reward_model = BinaryClassifier(encoder=RNNEncoder(V=V, D=D, H=400, L=1), enc_type='rnn', H=800)
            else:
                constant.bi = 'none'
                reward_model = BinaryClassifier(encoder=RNNEncoder(V=V, D=D, H=300, L=1), enc_type='rnn', H=300)
                constant.bi = 'bi'
        reward_model = load_model(reward_model, constant.reward_model)
        model.init_reward(reward_model)

        if constant.use_sentiment:
            # define and load sentiment model
            model.init_multitask()

            if constant.aux_reward_model != '':
                # define and load auxilary reward model
                reward_model = BinaryClassifier(encoder=BertModel.from_pretrained('bert-base-cased'), enc_type='bert', H=768)
                reward_model = load_model(reward_model, constant.aux_reward_model)
                model.init_aux_reward(reward_model)

        # create baseline reward layer
        if constant.use_baseline:
            model.init_baseline_reward()

        # create intrinsic curiosity module
        if constant.use_curiosity:
            model.init_intrinsic_curosity_module()

        train_fn = train_rl
    elif constant.task == 'lved':
        # supervised VAE 
        model = LVED(C=C, V=V, embedding=embedding)
        train_fn = train_lved
    else:
        print("Model is not defined")
        exit(1)

    print(model)
    if not constant.test:
        train_fn(model, dataloaders)
    # elif constant.eval:
    #     eval_fn(model, dataloader)
    else:
        test(model, dataloaders, constant.test_path)
