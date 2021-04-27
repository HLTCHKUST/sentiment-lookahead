import re

import numpy as np
import dill as pickle

import torch
import torch.utils.data as data
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils import constant, text_input2bert_input


class SentimentDataset(data.Dataset):
    def __init__(self, mode='train', dataset='sst', load_fasttext=False):
        self.mode = mode
        self.dataset = dataset
        self.fasttext = None
        self.load_fasttext = load_fasttext
        self.use_bert = constant.use_bert
        if self.use_bert:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self._from_file()

    def __len__(self):
        return self.sentiments.shape[0]

    def __getitem__(self, i):
        if self.use_bert:
            input_id, input_mask, segment_id = text_input2bert_input(self.texts[i], self.bert_tokenizer, seq_length=128)
            return input_id, input_mask, segment_id, self.sentiments[i]
        return self.sentences[i], self.sentence_lens[i], self.sentiments[i]

    def _from_file(self):
        def load_npy(path):
            return np.load(path)

        load_path = 'data/prep/{}/{}.{}.npy'

        self.sentiments        = load_npy(load_path.format(self.dataset, 'sentiments', self.mode))
        
        if self.use_bert:
            self.texts             = load_npy(load_path.format(self.dataset, 'texts', self.mode))
        else:
            self.sentences         = load_npy(load_path.format(self.dataset, 'sentences', self.mode))
            self.sentence_lens     = load_npy(load_path.format(self.dataset, 'sentence_lens', self.mode))
        
            if self.mode == 'train':
                if self.load_fasttext:
                    self.fasttext = load_npy('data/prep/{}/fasttext.npy'.format(self.dataset))

                with open('data/prep/{}/lang.pkl'.format(self.dataset), 'rb') as f:
                    self.lang = pickle.load(f)



def make_sentiment_data_loader(dataset, cuda, batch_size, pad_idx=1, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=collate_fn(cuda=cuda, bert=dataset.use_bert, pad_idx=pad_idx))


def collate_fn(cuda=False, bert=False, pad_idx=1):
    def collate_inner(batch):
        """
        Input
        - batch[0]: sentences -> [B x L]
        - batch[1]: sentence_lens -> [B]
        - batch[2]: sentiments -> [B]
        """
        # Unzip data (returns tuple of batches)
        if bert:
            input_ids, input_masks, segment_ids, sentiments = zip(*batch)
            input_ids = torch.stack(input_ids)
            input_masks = torch.stack(input_masks)
            segment_ids = torch.stack(segment_ids)
            sentiments = torch.from_numpy(np.array(sentiments)).float()
            if cuda:
                input_ids = input_ids.cuda()
                input_masks = input_masks.cuda()
                segment_ids = segment_ids.cuda()
                sentiments = sentiments.cuda()
            return input_ids, input_masks, segment_ids, sentiments
        else:
            sentences, sentence_lens, sentiments = zip(*batch)

        sort = np.argsort(sentence_lens)[::-1].tolist()
        sentences = np.array(sentences, dtype='object')[sort].tolist()
        sentence_lens = np.array(sentence_lens)[sort]#.tolist()
        sentiments = torch.from_numpy(np.array(sentiments)[sort]).float()

        # Pad dialogs and targets to their respective max batch lens
        B = len(sentences)
        L = sentence_lens[0]
        if pad_idx == 0:
            padded_sentences = torch.zeros((B, L))
        else:
            padded_sentences = torch.ones((B, L)) * pad_idx
        for b in range(B):
            padded_sentences[b, :sentence_lens[b]] = torch.from_numpy(np.array(sentences[b]))
      
        padded_sentences = padded_sentences.long()

        if cuda:
            padded_sentences = padded_sentences.cuda()
            sentiments = sentiments.cuda()

        return padded_sentences, sentence_lens, sentiments
    return collate_inner
