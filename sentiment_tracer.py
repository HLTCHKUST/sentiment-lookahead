import os
import math
import random

import dill as pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from models import RNNEncoder, BinaryClassifier

from train_sentiment import train_trace, train_sentiment

from utils import constant, text_input2bert_input
from utils.utils import load_ckpt, load_model


class TraceDataset(torch.utils.data.Dataset):
    def __init__(self, mode='targets', split='train', use_binary=False):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.texts = load_npy('data/prep/empathetic-dialogue/trc_{}.{}'.format(mode, split))
        if mode == 'dialogs':
            self.texts = [text.split('LISTENER: ')[1] for text in self.texts]
            if use_binary:
                self.traces = load_npy('data/prep/empathetic-dialogue/traces_binary.{}'.format(split))
            else:
                self.traces = load_npy('data/prep/empathetic-dialogue/traces.{}'.format(split))
        self.mode = mode
        self.split = split

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        input_id, input_mask, segment_id = text_input2bert_input(self.texts[i], self.bert_tokenizer, seq_length=128)
        if self.mode == 'dialogs':
            return input_id, input_mask, segment_id, self.traces[i]
        else:
            return input_id, input_mask, segment_id

class TraceDatasetRNN(torch.utils.data.Dataset):
    def __init__(self, split='train', use_binary=False):
        self.texts = load_npy('data/prep/empathetic-dialogue/usr_dialogs.{}'.format(split))
        # self.texts = [text.split('LISTENER: ')[1] for text in self.texts]
        self.lens = load_npy('data/prep/empathetic-dialogue/usr_dialog_lens.{}'.format(split))
        self.split = split
        if use_binary:
            self.traces = load_npy('data/prep/empathetic-dialogue/traces_binary.{}'.format(split))
        else:
            self.traces = load_npy('data/prep/empathetic-dialogue/sentiments_improved.{}'.format(split))

        self.fasttext = load_npy('data/prep/empathetic-dialogue/fasttext')
        with open('data/prep/empathetic-dialogue/lang.pkl', 'rb') as f:
            self.lang = pickle.load(f)
            
    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        return self.texts[i], self.lens[i], self.traces[i]


def collate_fn(cuda=False, use_trace=False):
    def collate_bert(batch):
        # Unzip data (returns tuple of batches)
        traces = None
        if use_trace:
            input_ids, input_masks, segment_ids, traces = zip(*batch)
            traces = torch.from_numpy(np.array(traces)).float()
        else:
            input_ids, input_masks, segment_ids = zip(*batch)
        input_ids = torch.stack(input_ids)
        input_masks = torch.stack(input_masks)
        segment_ids = torch.stack(segment_ids)

        if cuda:
            input_ids = input_ids.cuda()
            input_masks = input_masks.cuda()
            segment_ids = segment_ids.cuda()
            if use_trace:
                traces = traces.cuda()
        return input_ids, input_masks, segment_ids, traces

    def collate_rnn(batch):
        """
        Input
        - batch[0]: sentences -> [B x L]
        - batch[1]: sentence_lens -> [B]
        - batch[2]: sentiments -> [B]
        """
        # Unzip data (returns tuple of batches)
        sentences, sentence_lens, sentiments = zip(*batch)

        sort = np.argsort(sentence_lens)[::-1].tolist()
        sentences = np.array(sentences, dtype='object')[sort].tolist()
        sentence_lens = np.array(sentence_lens)[sort]#.tolist()
        sentiments = torch.from_numpy(np.array(sentiments)[sort]).float()

        # Pad dialogs and targets to their respective max batch lens
        B = len(sentences)
        L = sentence_lens[0]
        padded_sentences = torch.ones((B, L)) * constant.pad_idx
        for b in range(B):
            padded_sentences[b, :sentence_lens[b]] = torch.from_numpy(np.array(sentences[b]))
      
        padded_sentences = padded_sentences.long()

        if cuda:
            padded_sentences = padded_sentences.cuda()
            sentiments = sentiments.cuda()

        return padded_sentences, sentence_lens, sentiments

    if constant.use_bert:
        return collate_bert
    else:
        return collate_rnn

def make_data_loader(dataset, cuda, batch_size, shuffle=True, use_trace=False):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=collate_fn(cuda=cuda, use_trace=use_trace))

def save_npy(data, path):
    np.save(path+'.npy', data)

def load_npy(path):
    return np.load(path+'.npy')

def save_pkl(data, path):
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_pkl(path):
    with open(path+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    if not os.path.exists('data/prep/empathetic-dialogue/traces.{}.npy'.format(constant.split)):
        # 1. create dataloader for sentiment model
        dataset = TraceDataset(mode='targets', split=constant.split)
        dataloader = make_data_loader(dataset, constant.USE_CUDA, constant.B, shuffle=False, use_trace=False)

        # 2. define model
        encoder = BertModel.from_pretrained('bert-base-cased')
        model = BinaryClassifier(encoder=encoder, enc_type='bert', H=constant.H)

        # 3. load fine-tuned BERT model on SST + ED
        model = load_model(model, constant.test_path)
        if constant.USE_CUDA:
            model.cuda()
        model.eval()

        # 4. feed trace targets by batch, in order, and start labeling them
        pbar = tqdm(enumerate(dataloader),total=len(dataloader))

        preds = []
        probs = []
        sigmoid = nn.Sigmoid()
        with torch.no_grad():
            for _, batch in pbar:
                input_ids, input_masks, segment_ids, _ = batch
                logits = model((input_ids, segment_ids, input_masks)).squeeze()
                prob = sigmoid(logits)
                pred = sigmoid(logits) > 0.5
                probs.append(prob.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())

        probs = np.concatenate(probs)
        preds = np.concatenate(preds)
        print(probs.shape)
        print(preds.shape)

        # 5. Save the labeled as traces.{}.npy
        save_npy(probs, 'data/prep/empathetic-dialogue/traces.{}'.format(constant.split))
        save_npy(preds, 'data/prep/empathetic-dialogue/traces_binary.{}'.format(constant.split))

    else:
        # 1. create dataloaders for sentiment model
        if constant.use_bert:
            train_dataset = TraceDataset(mode='dialogs', split='train', use_binary=constant.use_binary)
            dev_dataset = TraceDataset(mode='dialogs', split='dev', use_binary=constant.use_binary)
            test_dataset = TraceDataset(mode='dialogs', split='test', use_binary=constant.use_binary)
        else:
            train_dataset = TraceDatasetRNN(split='train', use_binary=constant.use_binary)
            dev_dataset = TraceDatasetRNN(split='dev', use_binary=constant.use_binary)
            test_dataset = TraceDatasetRNN(split='test', use_binary=constant.use_binary)
            
        train_dataloader = make_data_loader(train_dataset, constant.USE_CUDA, constant.B, shuffle=constant.shuffle, use_trace=True)
        dev_dataloader = make_data_loader(dev_dataset, constant.USE_CUDA, constant.B, shuffle=constant.shuffle, use_trace=True)
        test_dataloader = make_data_loader(test_dataset, constant.USE_CUDA, constant.B, shuffle=constant.shuffle, use_trace=True)
        dataloaders = (train_dataloader, dev_dataloader, test_dataloader)

        # 2. define model
        if constant.use_bert:
            encoder = BertModel.from_pretrained('bert-base-cased')
            model = BinaryClassifier(encoder=encoder, enc_type='bert', H=constant.H)
        else:
            C = constant.C
            H = constant.H
            D = constant.D
            V = len(train_dataset.lang) 
            embedding = nn.Embedding(V, D)
            if constant.embedding == 'fasttext':
                embedding.weight = nn.Parameter(torch.from_numpy(train_dataset.fasttext).float())
                embedding.weight.requires_grad = constant.update_embeddings

            encoder = RNNEncoder(V=V, D=D, H=H, L=1, embedding=embedding)
            if constant.bi == 'bi':
                H = H * 2
            model = BinaryClassifier(encoder=encoder, enc_type='rnn', H=H)

        # 3. load fine-tuned BERT model on SST + ED
        if constant.test_path != "":
            model = load_model(model, constant.test_path)
            if constant.reset_linear:
                model.out = nn.Linear(model.H, 1)

        # 4. train reward model (linear regression) to predict sentiment score of next utterance
        if constant.use_bert:
            train_trace(model, dataloaders)
        else:
            train_sentiment(model, dataloaders)