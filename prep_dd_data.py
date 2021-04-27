import io
import re
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import dill as pickle

from utils.lang import Lang
from utils import constant


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

def create_vocab(path, lang):
    with open(path, 'r') as f:
        for line in f:
            line = re.sub(r'([a-zA-Z])([.])([A-Z])', r'\1 \2 \3', line)
            lang.addSentence(line.rstrip('\n'))

def read_dialog(path, lang, use_lang=True):
    full_dialogs = []
    flat_dialogs = []
    sld_dialogs = []
    sld_targets = []
    targets = []
    with open(path, 'r') as f:
        for line in f:
            line = re.sub(r'([a-zA-Z])([.])([A-Z])', r'\1 \2 \3', line)
            full_dialog = list(filter(lambda x: len(x) > 0, line.rstrip('\n').split('__eou__')))
            full_dialogs.append(" ".join(full_dialog))
            flat_dialogs.append(" ".join(full_dialog[:-1]))
            targets.append(full_dialog[-1] + ' __eou__')
            if use_lang:
                lang.addSentence(full_dialogs[-1])
            
            # Split dialogs by sliding context window
            for t in range(1, len(full_dialog)-1):
                sld_dialogs.append(" ".join(full_dialog[t-1:t+1]))# + ' __eou__')
                sld_targets.append(full_dialog[t+1] + ' __eou__')

    return full_dialogs, flat_dialogs, targets, sld_dialogs, sld_targets

def read_dialog_emo(path):
    # { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
    # => 0: happy, 1: angry, 2: sad, 3: others
    label_map = {
        0: 'none',
        1: 'anger',
        2: 'disgust',
        3: 'fear',
        4: 'happiness',
        5: 'sadness',
        6: 'surprise'
    }
    save_path = 'data/prep/dailydialog/{}' # name
    save_pkl(label_map, save_path.format('label_map'))

    # full_emotions = []

    curs = [] # emotional state at t
    tgts = [] # emotional state at t+1
    as_curs = [] # aug/sld
    as_tgts = []

    with open(path, 'r') as f:
        for line in f:
            emotions = list(filter(lambda x: len(x) > 0, line.rstrip('\n').split(' ')))
            emotions = [int(e) for e in emotions]
            turns = len(emotions)
            cur, tgt = emotions[-2:]
            curs.append(cur)
            tgts.append(tgt)
            for t in range(0, turns-1):
                as_cur, as_tgt = emotions[t], emotions[t+1]
                as_curs.append(as_cur)
                as_tgts.append(as_tgt)

    return curs, tgts, as_curs, as_tgts

def read_dialog_topic(path):
    # {1: Ordinary Life, 2: School Life, 3: Culture & Education, 4: Attitude & Emotion, 
    #  5: Relationship, 6: Tourism , 7: Health, 8: Work, 9: Politics, 10: Finance}    
    dialog_path = 'data/raw/dailydialog/dialogues_text.txt'
    topic_path = 'data/raw/dailydialog/dialogues_topic.txt'

    dialogs = []
    with open(dialog_path, 'r') as f:
        for line in f:
            dialogs.append(line)

    # {dialog: topic}
    topics = []
    with open(topic_path, 'r') as f:
        for line in f:
            topics.append(int(line) - 1)

    # hash dialogs
    dialogs = { d: t for d, t in zip(dialogs, topics) }

    topics = []
    as_topics = []
    with open(path, 'r') as f:
        for line in f:
            topics.append(dialogs[line])
            line = re.sub(r'([a-zA-Z])([.])([A-Z])', r'\1 \2 \3', line)
            full_dialog = list(filter(lambda x: len(x) > 0, line.rstrip('\n').split('__eou__')))
            # Split dialogs by sliding window or augmeting
            turns = len(full_dialog)
            as_topics += [topics[-1]] * turns
    
    return topics, as_topics
    
def read_dialog_act(path):
    # { 1: inform，2: question, 3: directive, 4: commissive }
    return 1
    
def transform_data(lang, flat_dialogs, targets, sld_dialogs, sld_targets):
    flat_dialogs = [lang.transform_one(dialog) for dialog in flat_dialogs]
    targets = [lang.transform_one(target) for target in targets]
    sld_dialogs = [lang.transform_one(dialog) for dialog in sld_dialogs]
    sld_targets = [lang.transform_one(target) for target in sld_targets]
    return flat_dialogs, targets, sld_dialogs, sld_targets

def get_lens(flat_dialogs, targets, sld_dialogs, sld_targets):
    # full_dialog_turns = [len(dialog) for dialog in full_dialogs]
    flat_dialog_lens = [len(dialog) for dialog in flat_dialogs]
    target_lens = [len(target) for target in targets]
    sld_dialog_lens = [len(dialog) for dialog in sld_dialogs]
    sld_target_lens = [len(target) for target in sld_targets]
    return flat_dialog_lens, target_lens, sld_dialog_lens, sld_target_lens

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


if __name__ == "__main__":
    """
    Preprocess DailyDialog data
    - Create vocab with Lang (lang.pkl)
    - Read dialogs line by line (full_dialogs.npy)
    - Read dialogs line by line (flat_dialogs.npy)
    - Split each dialog by '__eou__' for hierachical context (dialogs.npy)
    - Save last utterance + '__eou__' as targets (targets.npy)
    - Load emotions line by line and save last utterance emotions (emotions.npy)
    """
    if not os.path.exists('data/prep/dailydialog/lang.pkl'):
        print("Creating vocab from full text")
        lang = Lang()
        # create_vocab('data/raw/dailydialog/dialogues_text.txt', lang)
        if constant.use_lang:
            save_path = 'data/prep/dailydialog/{}'
            save_pkl(lang, save_path.format('lang'))
        print('lang before', len(lang))
    else:
        print("Loading vocab")
        lang = load_pkl('data/prep/dailydialog/lang')
        print('lang loaded', len(lang))

    if constant.embedding == 'fasttext' and not os.path.exists('data/prep/dailydialog/fasttext.npy'):
        save_path = 'data/prep/dailydialog/{}' # name
        embeddings = load_vectors('./vectors/crawl-300d-2M-subword.vec', lang.word2index)
        print("saving embeddings to", save_path.format('fasttext'))
        print(embeddings.shape)
        save_npy(embeddings, save_path.format('fasttext'))


    # Extract Emotions
    # if not os.path.exists('data/prep/dailydialog/tgt_emotions.{}.txt'.format(constant.split)):
    #     print('Preprocessing Emotions')
    #     load_path = 'data/raw/dailydialog/{}/dialogues{}_{}.txt'
    #     curs, tgts, as_curs, as_tgts = read_dialog_emo(load_path.format(constant.split, '_emotion', constant.split))
    #     for i in range(0, 8):
    #         print(curs[i], tgts[i])
    #     for i in range(0, 8):
    #         print(as_curs[i], as_tgts[i])
    #     save_path = 'data/prep/dailydialog/{}.{}' # name.split
    #     save_npy(curs, save_path.format('cur_emotions', constant.split))
    #     save_npy(tgts, save_path.format('tgt_emotions', constant.split))
    #     save_npy(as_curs, save_path.format('as_cur_emotions', constant.split))
    #     save_npy(as_tgts, save_path.format('as_tgt_emotions', constant.split))


    # Extract Topics
    # for s in ['train', 'dev', 'test']:
    # s = constant.split
    # if not os.path.exists('data/prep/dailydialog/topic_{}.npy'.format(s)):
    #     print('Preprocessing Topics')
    #     load_path = 'data/raw/dailydialog/{}/dialogues_{}.txt'.format(s, s)
    #     topics, as_topics = read_dialog_topic(load_path)
    #     print(len(topics), len(as_topics))
    #     save_path = 'data/prep/dailydialog/{}.{}' # name.split
    #     save_npy(topics, save_path.format('topics', s))
    #     save_npy(as_topics, save_path.format('as_topics', s))
    

    # Extract Dialogs
    load_path = 'data/raw/dailydialog/{}/dialogues{}_{}.txt'
    full_dialogs, flat_dialogs, targets, sld_dialogs, sld_targets = read_dialog(load_path.format(constant.split, '', constant.split), lang, use_lang=constant.use_lang)
    if constant.use_lang:
        print('lang after', len(lang))
        save_path = 'data/prep/dailydialog/{}'
        save_pkl(lang, save_path.format('lang'))
    print("Flat", flat_dialogs[0])
    print("Targets", targets[0])
    print("Sld Dial", sld_dialogs[0])
    print("A/S Target", sld_targets[0])
    print("Sld Dial", sld_dialogs[1])
    print("A/S Target", sld_targets[1])
    print("Sld Dial", sld_dialogs[2])
    print("A/S Target", sld_targets[2])
    print(lang.index2word[3])
    print(lang.word2index['__eou__'])
    print(lang.index2word[1])
    print(lang.word2index['__pad__'])
    print(lang.word2index['.'])
    save_path = 'data/prep/dailydialog/{}.{}'
    save_npy(full_dialogs, save_path.format('full_dialog_texts', constant.split))
    save_npy(sld_dialogs, save_path.format('dialog_texts', constant.split))
    save_npy(sld_targets, save_path.format('target_texts', constant.split))


    flat_dialogs, targets, sld_dialogs, sld_targets = transform_data(lang, flat_dialogs, targets, sld_dialogs, sld_targets)
    print("Flat", flat_dialogs[0])
    print("Targets", targets[0])
    print("Sld Dial", sld_dialogs[0])
    print("A/S Target", sld_targets[0])
    print("Sld Dial", sld_dialogs[1])
    print("A/S Target", sld_targets[1])
    print("Sld Dial", sld_dialogs[2])
    print("A/S Target", sld_targets[2])

    save_path = 'data/prep/dailydialog/{}.{}'
    save_npy(sld_dialogs, save_path.format('dialogs', constant.split))
    save_npy(sld_targets, save_path.format('targets', constant.split))

    flat_dialog_lens, target_lens, sld_dialog_lens, sld_target_lens = get_lens(flat_dialogs, targets, sld_dialogs, sld_targets)
    save_npy(sld_dialog_lens, save_path.format('dialog_lens', constant.split))
    save_npy(sld_target_lens, save_path.format('target_lens', constant.split))
