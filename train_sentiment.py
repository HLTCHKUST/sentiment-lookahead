import os
import math
import random

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam

from utils import constant, masked_cross_entropy
from utils.bleu import moses_multi_bleu
from utils.utils import get_metrics, save_ckpt, load_ckpt, save_model, load_model


def train_trace(model, dataloaders):
    train_dataloader, dev_dataloader, test_dataloader = dataloaders
    if(constant.USE_CUDA): model.cuda()

    if constant.use_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = BertAdam(optimizer_grouped_parameters,
                    lr=constant.lr,
                    warmup=0.01,
                    t_total=int(len(train_dataloader) * 5))

    best_dev = 10000
    best_test = 10000
    patience = 3

    for e in range(constant.epochs):
        model.train()
        loss_log = []
        f1_log = []

        pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
    
        for _, batch in pbar:
            input_ids, input_masks, segment_ids, traces = batch
            logits = model((input_ids, segment_ids, input_masks)).squeeze()

            if len(logits.shape) == 0:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, traces)
            loss.backward()
            opt.step()
            opt.zero_grad()

            ## logging
            loss_log.append(loss.item())
            if constant.use_binary:
                preds = F.sigmoid(logits) > 0.5
                golds = traces.cpu().numpy()
            else:
                preds = logits > 0.5
                golds = (traces > 0.5).cpu().numpy()
            f1 = f1_score(golds, preds.detach().cpu().numpy(), average='weighted')
            f1_log.append(f1)

            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} TRAIN F1:{:.4f}".format(e+1, np.mean(loss_log), np.mean(f1_log)))
        
        ## LOG
        dev_loss, dev_f1 = eval_trace(model, dev_dataloader)
        test_loss, test_f1 = eval_trace(model, test_dataloader)
        print("(Epoch {}) DEV LOSS: {:.4f} DEV F1:{:.4f} TEST LOSS: {:.4f} TEST F1:{:.4f} ".format(e+1, dev_loss, dev_f1, test_loss, test_f1))
        print("(Epoch {}) BEST DEV LOSS: {:.4f} BEST TEST LOSS: {:.4f}".format(e+1, best_dev, best_test))
        if(dev_loss < best_dev):
            best_dev = dev_loss
            best_test = test_loss
            patience = 3
            path = 'trained/data-{}.task-trace.loss-{}'
            save_model(model, 'loss', best_dev, path.format(constant.data, best_dev))
        else:
            patience -= 1
        if(patience == 0): break
        if(best_dev == 0.0): break 

    print("BEST SCORES - DEV LOSS: {:.4f}, TEST LOSS: {:.4f}".format(best_dev, best_test))


def eval_trace(model, dataloader):
    model.eval()
    if constant.use_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    loss_log = []
    f1_log = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, input_masks, segment_ids, traces = batch
            logits = model((input_ids, segment_ids, input_masks)).squeeze()
            if len(logits.shape) == 0:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, traces)
            loss_log.append(loss.item())

            if constant.use_binary:
                preds = F.sigmoid(logits) > 0.5
                golds = traces.cpu().numpy()
            else:
                preds = logits > 0.5
                golds = (traces > 0.5).cpu().numpy()

            f1 = f1_score(golds, preds.detach().cpu().numpy(), average='weighted')
            f1_log.append(f1)

    return np.mean(loss_log), np.mean(f1_log)


def train_sentiment(model, dataloaders):
    """ 
    Training loop
    Inputs:
        model: the model to be trained
        dataloader: data loader
    Output:
        best_dev: best f1 score on dev data
        best_test: best f1 score on test data
    """
    train_dataloader, dev_dataloader, test_dataloader = dataloaders
    if(constant.USE_CUDA): model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    
    if constant.use_bert:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        opt = BertAdam(optimizer_grouped_parameters,
                        lr=constant.lr,
                        warmup=0.01,
                        t_total=int(len(train_dataloader) * 5))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=constant.lr)

    best_dev = 0
    best_test = 0
    patience = 3

    try:
        for e in range(constant.epochs):
            model.train()
            loss_log = []
            f1_log = []

            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        
            if constant.grid_search:
                pbar = enumerate(train_dataloader)
            else:
                pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))

            for _, batch in pbar:
                if constant.use_bert:
                    input_ids, input_masks, segment_ids, sentiments = batch
                    logits = model((input_ids, segment_ids, input_masks)).squeeze()
                else:
                    sentences, lens, sentiments = batch
                    logits = model(sentences, lens).squeeze()

                if len(logits.shape) == 0:
                    logits = logits.unsqueeze(0)
                loss = criterion(logits, sentiments)
                loss.backward()
                opt.step()
                opt.zero_grad()

                ## logging 
                loss_log.append(loss.item())
                preds = F.sigmoid(logits) > 0.5
                # preds = torch.argmax(logits, dim=1)
                f1 = f1_score(sentiments.cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
                f1_log.append(f1)
                if not constant.grid_search:
                    pbar.set_description("(Epoch {}) TRAIN F1:{:.4f} TRAIN LOSS:{:.4f}".format(e+1, np.mean(f1_log), np.mean(loss_log)))
            
            ## LOG
            f1 = eval_sentiment(model, dev_dataloader)
            testF1 = eval_sentiment(model, test_dataloader)
            print("(Epoch {}) DEV F1: {:.4f} TEST F1: {:.4f}".format(e+1, f1, testF1))
            print("(Epoch {}) BEST DEV F1: {:.4f} BEST TEST F1: {:.4f}".format(e+1, best_dev, best_test))
            if(f1 > best_dev):
                best_dev = f1
                best_test = testF1
                patience = 3
                path = 'trained/data-{}.task-sentiment.f1-{}'
                save_model(model, 'loss', best_dev, path.format(constant.data, best_dev))
            else:
                patience -= 1
            if(patience == 0): break
            if(best_dev == 1.0): break 

    except KeyboardInterrupt:
        if not constant.grid_search:
            print("KEYBOARD INTERRUPT: Save CKPT and Eval")
            save = True if input('Save ckpt? (y/n)\t') in ['y', 'Y', 'yes', 'Yes'] else False
            if save:
                save_path = save_ckpt(model, opt, e)
                print("Saved CKPT path: ", save_path)
            print("BEST SCORES - DEV F1: {:.4f}, TEST F1: {:.4f}".format(best_dev, best_test))
        exit(1)

    print("BEST SCORES - DEV F1: {:.4f}, TEST F1: {:.4f}".format(best_dev, best_test))


def eval_sentiment(model, dataloader):
    model.eval()
    preds = []
    golds = []
    with torch.no_grad():
        for batch in dataloader:
            if constant.use_bert:
                input_ids, input_masks, segment_ids, sentiments = batch
                logits = model((input_ids, segment_ids, input_masks)).squeeze()
            else:
                sentences, lens, sentiments = batch
                logits = model(sentences, lens).squeeze()
            pred = logits > 0.5
            preds.append(pred.detach().cpu().numpy())
            golds.append(sentiments.cpu().numpy())

    preds = np.concatenate(preds)
    golds = np.concatenate(golds)
    f1 = f1_score(golds, preds, average='weighted')
    # _, _, _, microF1 = get_metrics(pred, gold, verbose=False if constant.grid_search else True)
    return f1
