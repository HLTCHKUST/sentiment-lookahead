#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which iterates through the tasks specified and prints them out.
Used for verification of data loading and iteration.
For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:
Examples
--------
.. code-block:: shell
  python display_data.py -t babi:task1k:1
"""

import io
import re
import os
import sys
import json

import numpy as np
import pandas as pd
import dill as pickle

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import nltk
import spacy

class Lang:
    def __init__(self):
        self.unk_idx = 0
        self.pad_idx = 1
        self.sou_idx = 2
        self.eou_idx = 3

        self.word2index = {'__unk__': self.unk_idx, '__pad__': self.pad_idx, '__sou__': self.sou_idx, '__eou__': self.eou_idx}
        self.word2count = {'__unk__': 0, '__pad__': 0, '__sou__': 0, '__eou__': 0}
        self.index2word = {self.unk_idx: "__unk__", self.pad_idx: "__pad__", self.sou_idx: "__sou__", self.eou_idx: "__eou__"} 
        self.n_words = 4 # Count default tokens

        self.nlp = spacy.load("en_core_web_sm")
        # add special case rule
        special_case = [{spacy.symbols.ORTH: u"__eou__"}]
        self.nlp.tokenizer.add_special_case(u"__eou__", special_case)

    def __len__(self):
        return len(self.word2index)

    def tokenize(self, s):
        # return nltk.word_tokenize(s)
        return self.nlp.tokenizer(s)

    def addSentence(self, sentence):
        # for word in sentence.split(' '):
        for word in self.tokenize(sentence):
            self.addWord(word.text)

    def addSentences(self, sentences):
        for sentence in sentences:
            for word in self.tokenize(sentence):
                self.addWord(word.text)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def transform(self, sentences):
        # given unokenized sentences (or iterator), transform to idx mapping
        return [[self.word2index[token.text] for token in self.tokenize(sentence)] for sentence in sentences]

    def transform_one(self, sentence):
        try:
        # given unokenized sentence, transform to idx mapping
            return [self.word2index[token.text] for token in self.tokenize(sentence)]
        except KeyError as e:
            print(e)
            print(sentence)
            return []

    def reverse(self, sentences):
        # given transformed sentences, reverse it
        return [[self.index2word[idx] for idx in sentence] for sentence in sentences]

    def reverse_one(self, sentence):
        # given transformed sentence, reverse it
        return [self.index2word[idx] for idx in sentence]


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Display data from a task')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-ne', '--num_examples', type=int, default=10)
    parser.add_argument('-mdl', '--max_display_len', type=int, default=1000)
    parser.add_argument('--display_ignore_fields', type=str, default='agent_reply')
    parser.set_defaults(datatype='train:stream')
    return parser

def clean_msg(msg, msg_type):
    if 'labels' not in msg_type:
        msg = msg[3+len(msg_type):].strip()
    else:
        msg = msg[2+len(msg_type):-1].strip()
    msg = re.sub(r'"', '', msg)
    return msg

def display_data(opt, lang):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    dialog = []
    full_dialogs = []
    dialogs = [] #context window of two
    targets = []
    for i in range(opt['num_examples']):
        world.parley()

        # NOTE: If you want to look at the data from here rather than calling
        # world.display() you could access world.acts[0] directly
        message = world.display().split('\n')

        for i, line in enumerate(message):
            # new dialog sequence
            if i == 0 and '[personachat]: your persona:' in line:
                # dialogs.append(flatten(dialog[:-1]))
                # Split dialogs by sliding context window
                full_dialogs.append(" ".join(dialog))
                add_sentence(lang, full_dialogs[-1])
                for t in range(1, len(dialog)-1):
                    dialogs.append(" ".join(dialog[t-1:t+1]))
                    targets.append(dialog[t+1] + ' __eou__')
                dialog = []
            elif i > 0 and 'your persona:' in line:
                continue
            elif '[label_candidates:' in line:
                continue
            elif '[labels:' in line or '[eval_labels:' in line:
                if 'train' not in opt['datatype']:
                    dialog.append(clean_msg(line, 'eval_labels'))
                else:
                    dialog.append(clean_msg(line, 'labels'))
            elif '[personachat]:' in line and 'your persona:' not in line:
                dialog.append(clean_msg(line, 'personachat'))
            else:
                if line != '- - - - - - - - - - - - - - - - - - - - -':
                    dialog.append(re.sub(r'"', '', line.strip()))

        if world.epoch_done():
            print('EPOCH DONE')
            break

    try:
        # print dataset size if available
        print('[ loaded {} episodes with a total of {} examples ]'.format(
            world.num_episodes(), world.num_examples()
        ))
    except Exception:
        pass
        
    return full_dialogs, dialogs, targets

def transform_data(lang, dialogs, targets):
    dialogs = [lang.transform_one(sentence) for sentence in dialogs]
    targets = [lang.transform_one(sentence) for sentence in targets]
    dialog_lens = [len(sentence) for sentence in dialogs]
    target_lens = [len(sentence) for sentence in targets]
    return dialogs, targets, dialog_lens, target_lens

def flatten(dialog):
    return ' '.join(dialog)

def add_sentence(lang, sent):
    lang.addSentence(sent)

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

def load_vectors(fname, vocabulary):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embedding_matrix = np.random.uniform(-0.01, 0.01, ((len(vocabulary)), d))
    print(embedding_matrix.shape)
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocabulary.keys():
            embedding_matrix[vocabulary[tokens[0]]] = np.array(list(map(float, tokens[1:])))
    print(embedding_matrix.shape)

    return embedding_matrix


if __name__ == '__main__':
    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists('data/prep/personachat/lang.pkl'):
        print("Creating vocab from full text")
        lang = Lang()
        print('lang before', len(lang))
    else:
        print("Loading vocab")
        sys.argv = ['']
        lang = load_pkl('data/prep/personachat/lang')
        print('lang loaded', len(lang))

    full_dialogs, dialogs, targets = display_data(opt, lang)
    print(len(dialogs), len(targets))
    print('lang after', len(lang))
    print("Dial", dialogs[0])
    print("Target", targets[0])
    print("Dial", dialogs[1])
    print("Target", targets[1])
    print("Dial", dialogs[2])
    print("Target", targets[2])

    save_path = 'data/prep/personachat/{}'
    save_pkl(lang, save_path.format('lang'))
    split = re.sub(r':ordered', '', opt['datatype'])
    split = re.sub(r':stream', '', split)
    split = re.sub(r'valid', 'dev', split)
    print(split)
    save_path = 'data/prep/personachat/{}.{}' # name
    save_npy(full_dialogs, save_path.format('full_dialog_texts', split))
    save_npy(dialogs, save_path.format('dialog_texts', split))
    save_npy(targets, save_path.format('target_texts', split))

    dialogs, targets, dialog_lens, target_lens = transform_data(lang, dialogs, targets)
    print("Dial", dialogs[0])
    print("Target", targets[0])
    print("Dial", dialogs[1])
    print("Target", targets[1])
    print("Dial", dialogs[2])
    print("Target", targets[2])

    split = re.sub(r':ordered', '', opt['datatype'])
    split = re.sub(r':stream', '', split)
    split = re.sub(r'valid', 'dev', split)
    print(split)

    save_path = 'data/prep/personachat/{}.{}'
    save_npy(dialogs, save_path.format('dialogs', split))
    save_npy(targets, save_path.format('targets', split))
    save_npy(dialog_lens, save_path.format('dialog_lens', split))
    save_npy(target_lens, save_path.format('target_lens', split))