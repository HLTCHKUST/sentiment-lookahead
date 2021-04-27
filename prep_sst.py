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


def flatten_tree(tree):
    """
    Flattens constituency trees to get just the tokens.
    """
    label = int(tree[1])
    text = re.sub('\([0-9]', ' ', tree).replace(')', '').split()
    return label, ' '.join(text)

def read_data_sst_binary(file_name, lang, use_lang=True):
    X = []
    Y = []
    with open(file_name) as fl:
        for line in fl:
            y, x = flatten_tree(line)
            if use_lang:
                add_sentence(lang, x)
            if y in [0, 1]:
                X.append(x)
                Y.append(0)
            elif y in [3, 4]:
                X.append(x)
                Y.append(1)
    return X, Y

def add_sentence(lang, sent):
    lang.addSentence(sent)

def transform_data(lang, sentences):
    sentences = [lang.transform_one(sentence) for sentence in sentences]
    sentence_lens = [len(sentence) for sentence in sentences]
    return sentences, sentence_lens

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
    if not os.path.exists('data/prep/sst/lang.pkl'):
        print("Creating vocab from full text")
        lang = Lang()
        if constant.use_lang:
            save_path = 'data/prep/sst/{}'
            save_pkl(lang, save_path.format('lang'))
    else:
        print("Loading vocab")
        lang = load_pkl('data/prep/sst/lang')
    print('lang before', len(lang))

    # read data
    s = constant.split
    data_path = './data/raw/sst/{}.txt'.format(s)
    texts, sentiments = read_data_sst_binary(data_path, lang, use_lang=constant.use_lang)
    print(texts[0])
    print(sentiments[0])
    print('lang after', len(lang))
    if constant.use_lang:
        save_path = 'data/prep/sst/{}'
        save_pkl(lang, save_path.format('lang'))

    # transform data
    sentences, sentence_lens = transform_data(lang, texts)
    print(sentences[0])
    print(sentence_lens[0])

    if not os.path.exists('data/prep/sst/fasttext.npy') and constant.embedding == 'fasttext':
        embeddings = load_vectors('./vectors/crawl-300d-2M-subword.vec', lang.word2index)
        print("saving embeddings to", 'data/prep/sst/fasttext.npy')
        print(embeddings.shape)
        save_npy(embeddings, 'data/prep/sst/fasttext')

    save_path = 'data/prep/sst/{}.{}' # name.split
    save_npy(texts, save_path.format('texts', constant.split))
    save_npy(sentences, save_path.format('sentences', constant.split))
    save_npy(sentence_lens, save_path.format('sentence_lens', constant.split))
    save_npy(sentiments, save_path.format('sentiments', constant.split))
    