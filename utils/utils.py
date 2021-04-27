import math
from datetime import datetime
from functools import reduce
import operator

import numpy as np
from nltk.util import ngrams, everygrams

import torch

from utils import constant


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def save_model(model, metric, score, path=None):
    if path is not None:
        save_path = path
    else:
        save_path = 'trained/{}.{}.{}.{}.{}.{}.{:.4f}.{}' # data.task.model.H.lr.attn.metric.parse
        misc = ''
        if constant.lstm:
            misc += 'lstm.'
 
        save_path = save_path.format(constant.data, constant.task, constant.model, constant.H, constant.lr, constant.attn, metric, score, misc)
    torch.save(model.state_dict(), save_path)
    return save_path

def load_model(model, path):
    if path == "":
        return model
    if constant.USE_CUDA:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model
    
def save_ckpt(model, optim, epoch):
    save_path = 'ckpt/{}.{}.{}.{}.{}.{}' # dataset.task.epoch.lr.misc.time
    misc = ''
    if constant.lstm:
        misc += 'lstm.'
    date = datetime.now().date()
    time = datetime.now().time()
    dt = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, time.hour, time.minute, time.second)
    save_path = save_path.format(constant.data, constant.task, epoch, constant.lr, misc, dt)
        
    state = {
        'epoch': epoch, 
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()
    }
    torch.save(state, save_path)
    return save_path

def load_ckpt(model, optim, path):
    # Note: Input model & optimizer should be pre-defined.
    # This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        if constant.USE_CUDA:
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))

    return model, optim, start_epoch

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def get_metrics(predictions, ground, C=7, verbose=False):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    one_hot = np.zeros((ground.shape[0], C))
    one_hot[np.arange(ground.shape[0]), ground] = 1
    ground = one_hot
    label2emotion = {
        0: 'none',
        1: 'anger',
        2: 'disgust',
        3: 'fear',
        4: 'happiness',
        5: 'sadness',
        6: 'surprise'
    }
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1), num_classes=C)
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    if(verbose):
        print("True Positives per class : ", truePositives)
        print("False Positives per class : ", falsePositives)
        print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, C):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        if(verbose):
            print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    if(verbose):
        print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()
    if(verbose):
        print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    if(verbose):
        print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

def text_input2bert_input(text, bert_tokenizer, seq_length=512):
    tokens_a = bert_tokenizer.tokenize(text)
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = [] # equals raw text tokens 
    input_type_ids = [] # equals segments_ids
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens) # WordPiece embedding rep

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    input_ids_batch = torch.tensor(input_ids, dtype=torch.long)
    input_mask_batch = torch.tensor(input_mask, dtype=torch.long)
    segment_id_batch = torch.zeros(input_ids_batch.size(), dtype=torch.long)

    return input_ids_batch, input_mask_batch, segment_id_batch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits

def get_sentiment(sentiment_clf, sentences, tokenizer):
    input_ids, input_masks, segment_ids = zip(*[text_input2bert_input(sentence, tokenizer, seq_length=128) for sentence in sentences])
    input_ids = torch.stack(input_ids)
    input_masks = torch.stack(input_masks)
    segment_ids = torch.stack(segment_ids)

    if constant.USE_CUDA:
        input_ids = input_ids.cuda()
        input_masks = input_masks.cuda()
        segment_ids = segment_ids.cuda()

    # get reward with generated sentence
    with torch.no_grad():
        R = sentiment_clf.predict_prob((input_ids, segment_ids, input_masks))

    return R

def get_user_response(user_model, refs, sents, vocab):
    sents = [vocab.transform_one(sent) for sent in sents]
    lens = [len(sentence) for sentence in sents]
    sort = np.argsort(lens)[::-1].tolist()
    unsort = np.argsort(sort).tolist()
    sents = np.array(sents, dtype='object')[sort].tolist()
    lens = np.array(lens)[sort]

    B = len(sents)
    L = lens[0]
    padded_sents = torch.ones((B, L)) * constant.pad_idx
    for b in range(B):
        padded_sents[b, :lens[b]] = torch.from_numpy(np.array(sents[b]))

    padded_sents = padded_sents.long()
    if constant.USE_CUDA:
        padded_sents = padded_sents.cuda()

    return np.array(user_model.predict_batch(padded_sents, lens, np.zeros((B, L))))[unsort].tolist()

def distinct_ngrams(sentences):
    unigram = []
    bigram = []
    trigram = []
    for sent in sentences:
        s = sent.split()
        unigram.append(s)
        bigram.append(list(ngrams(s, 2)))
        trigram.append(list(ngrams(s, 3)))
    unigram = reduce(operator.concat, unigram)
    bigram = reduce(operator.concat, bigram)
    trigram = reduce(operator.concat, trigram)
    d1 = len(set(unigram))/len(unigram)
    d2 = len(set(bigram))/len(bigram)
    d3 = len(set(trigram))/len(trigram)
    return d1, d2, d3

# def get_embedding_similarity(refs, sents, vocab, encoder, mode='average', model='fasttext'):
#     if model == 'fasttext':
#         sents = [vocab.transform_one(sent) for sent in sents]
#         lens = np.array(lens)[sort]

#         B = len(sents)
#         L = lens[0]
#         padded_sents = torch.ones((B, L)) * constant.pad_idx
#         for b in range(B):
#             padded_sents[b, :lens[b]] = torch.from_numpy(np.array(sents[b]))

#         padded_sents = padded_sents.long()
#         if constant.USE_CUDA:
#             padded_sents = padded_sents.cuda()
#     elif model == 'bert':
#         pass
