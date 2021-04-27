import re

import numpy as np
import dill as pickle

import torch
import torch.utils.data as data

from utils import constant


class DialogDataset(data.Dataset):
    def __init__(self, mode='train', dataset='empathetic-dialogue', usr=False, sys=False, path=None, load_fasttext=False):
        self.mode = mode
        self.dataset = dataset
        self.usr = usr
        self.sys = sys
        self.fasttext = None
        self.load_fasttext = load_fasttext
        self.use_emotion = constant.use_emotion
        self.use_sentiment = constant.use_sentiment or constant.use_sentiment_agreement

        self._from_file(path)

    def __len__(self):
        if self.sys and self.dataset == 'empathetic-dialogue':
            return self.sys_target_lens.shape[0]
        elif self.usr and self.dataset == 'empathetic-dialogue':
            return self.usr_target_lens.shape[0]
        return self.target_lens.shape[0]

    def __getitem__(self, i):
        dialog = None
        dialog_len = None
        target = None
        target_len = None
        emotion = None
        sentiments = None
        if self.sys and self.dataset == 'empathetic-dialogue':
            dialog = self.sys_dialogs[i]
            dialog_len = self.sys_dialog_lens[i]
            target = self.sys_targets[i]
            target_len = self.sys_target_lens[i]
            if self.use_emotion:
                emotion = self.sys_emotions[i]
            elif self.use_sentiment:
                emotion = self.sys_sentiments[i]
                sentiments = self.sys_sentiments_b[i]
        elif self.usr and self.dataset == 'empathetic-dialogue':
            dialog = self.usr_dialogs[i]
            dialog_len = self.usr_dialog_lens[i]
            target = self.usr_targets[i]
            target_len = self.usr_target_lens[i]
            if self.use_emotion:
                emotion = self.sys_emotions[i]
            elif self.use_sentiment:
                emotion = self.sys_sentiments[i]
        else:
            dialog = self.dialogs[i]
            dialog_len = self.dialog_lens[i]
            target = self.targets[i]
            target_len = self.target_lens[i]
            if self.use_emotion:
                emotion = self.emotions[i]
            
        return dialog, dialog_len, target, target_len, emotion, sentiments

    def _from_file(self, path=None):
        def load_npy(path):
            return np.load(path)

        load_path = path if path else 'data/prep/{}/{}.{}.npy'
        
        if self.use_emotion:
            self.emotions            = load_npy(load_path.format(self.dataset, 'emotions', self.mode))

        if self.dataset == 'empathetic-dialogue':
            self.usr_dialogs         = load_npy(load_path.format(self.dataset, 'usr_dialogs', self.mode))
            self.usr_dialog_lens     = load_npy(load_path.format(self.dataset, 'usr_dialog_lens', self.mode))
            self.sys_dialogs         = load_npy(load_path.format(self.dataset, 'sys_dialogs', self.mode))
            self.sys_dialog_lens     = load_npy(load_path.format(self.dataset, 'sys_dialog_lens', self.mode))
            self.usr_targets         = load_npy(load_path.format(self.dataset, 'usr_targets', self.mode))
            self.usr_target_lens     = load_npy(load_path.format(self.dataset, 'usr_target_lens', self.mode))
            self.sys_targets         = load_npy(load_path.format(self.dataset, 'sys_targets', self.mode))
            self.sys_target_lens     = load_npy(load_path.format(self.dataset, 'sys_target_lens', self.mode))
            self.sys_emotions        = load_npy(load_path.format(self.dataset, 'sys_emotions', self.mode))
            self.sys_sentiments      = load_npy(load_path.format(self.dataset, 'sys_sentiments', self.mode))
            self.sys_sentiments_b    = load_npy(load_path.format(self.dataset, 'sys_sentiments_binary', self.mode))
        else:
            self.dialogs             = load_npy(load_path.format(self.dataset, 'dialogs', self.mode))
            self.dialog_lens         = load_npy(load_path.format(self.dataset, 'dialog_lens', self.mode))
            self.targets             = load_npy(load_path.format(self.dataset, 'targets', self.mode))
            self.target_lens         = load_npy(load_path.format(self.dataset, 'target_lens', self.mode))

        if self.mode == 'train':
            if self.load_fasttext:
                self.fasttext = load_npy('data/prep/{}/fasttext.npy'.format(self.dataset))

            with open('data/prep/{}/lang{}.pkl'.format(self.dataset, constant.lang_path), 'rb') as f:
                self.lang = pickle.load(f)


def make_dialog_data_loader(dataset, cuda, embeddings_cpu, batch_size, pad_idx=1, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=collate_fn(mode=dataset.mode, cuda=cuda, 
                                       embeddings_cpu=embeddings_cpu, pad_idx=pad_idx, V=len(dataset.lang), 
                                       use_emotion=dataset.use_emotion, use_sentiment=dataset.use_sentiment))


def collate_fn(mode='train', cuda=False, embeddings_cpu=False, pad_idx=1, V=None, use_emotion=False, use_sentiment=False):
    def collate_inner(batch):
        """
        Input
        - batch[0]: dialogs -> [B x UTT_LEN]
        - batch[1]: dialog_lens -> [B]
        - batch[2]: targets -> [B x TGT_LEN]
        - batch[3]: target_lens -> [B]
        - batch[4]: emotions -> [B]

        Returns
        - dialogs -> Ready for embedding lookup and packing
            - Original: Tensor of [B x MAX_TURN x MAX_UTT_LEN], padded with PAD words and PAD arrays
                Use pack_padded_sequence => Transform to [B * MAX_TURN x MAX_UTT_LEN] later for tensor computation
            - Flattened: Tensor of [B x MAX_SEQ_LEN], where MAX_SEQ_LEN is max flattened seq len in current batch
                Use pack_sequence
        - labels -> Tensor of [B] indicating index of correct emotion
        """

        # Unzip data (returns tuple of batches)
        dialogs, dialog_lens, targets, target_lens, emotions, sentiments = zip(*batch)

        sort = np.argsort(dialog_lens)[::-1].tolist()
        unsort = np.argsort(sort).tolist()
        dialogs = np.array(dialogs, dtype='object')[sort].tolist()
        lens = np.array(dialog_lens)[sort]#.tolist()
        targets = np.array(targets, dtype='object')[sort]#.tolist()
        target_lens = np.array(target_lens)[sort]#.tolist()

        bow_targets, x_sort, x_unsort = None, None, None
        # x_sort = np.argsort(target_lens)[::-1].tolist()
        # x_unsort = np.argsort(x_sort).tolist()
        # bow_targets = np.zeros((len(targets), V))
        # for i, target in enumerate(targets):
        #     bow_targets[i][target] = 1
        # bow_targets = torch.from_numpy(bow_targets).float()
        if use_emotion:
            emotions = torch.from_numpy(np.array(emotions)[sort]).long()
        elif use_sentiment:
            emotions = torch.from_numpy(np.array(emotions)[sort]).float()
            sentiments = torch.from_numpy(np.array(sentiments)[sort]).float()

        # Pad dialogs and targets to their respective max batch lens
        B = len(dialogs)
        LD = lens[0]
        LT = np.max(target_lens)
        if pad_idx == 0:
            padded_dialogs = torch.zeros((B, LD))
            padded_targets = torch.zeros((B, LT))
        else:
            padded_dialogs = torch.ones((B, LD)) * pad_idx
            padded_targets = torch.ones((B, LT)) * pad_idx
        for b in range(B):
            padded_dialogs[b, :lens[b]] = torch.from_numpy(np.array(dialogs[b]))
            padded_targets[b, :target_lens[b]] = torch.from_numpy(np.array(targets[b]))
      
        padded_dialogs = padded_dialogs.long()
        padded_targets = padded_targets.long()

        target_lens = torch.LongTensor(target_lens)
        if not embeddings_cpu and cuda:
            padded_dialogs = padded_dialogs.cuda()
            padded_targets = padded_targets.cuda()
            target_lens = target_lens.cuda()
            if use_emotion or use_sentiment:
                emotions = emotions.cuda()
                if use_sentiment:
                    sentiments = sentiments.cuda()

        return padded_dialogs, lens, padded_targets, unsort, bow_targets, emotions, sentiments, x_sort, x_unsort
    return collate_inner
