import os
import math
import random

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import constant, masked_cross_entropy
from utils.bleu import moses_multi_bleu
from utils.utils import get_metrics, save_ckpt, load_ckpt, save_model, load_model


def train_emotion(model, dataloaders):
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
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=constant.lr)

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

            for _, (dialogs, lens, _, _, emotions, _, _, _) in pbar:
                opt.zero_grad()
                logits = model(dialogs, lens)
                
                loss = criterion(logits, emotions)
                loss.backward()
                opt.step()

                ## logging 
                loss_log.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                f1 = f1_score(emotions.cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
                # _, _, _, microF1 = get_metrics(logits.detach().cpu().numpy(), emotions.cpu().numpy())
                f1_log.append(f1)
                if not constant.grid_search:
                    pbar.set_description("(Epoch {}) TRAIN F1:{:.4f} TRAIN LOSS:{:.4f}".format(e+1, np.mean(f1_log), np.mean(loss_log)))
            
            ## LOG
            f1 = eval_emotion(model, dev_dataloader)
            testF1 = eval_emotion(model, test_dataloader)
            print("(Epoch {}) DEV F1: {:.4f} TEST F1: {:.4f}".format(e+1, f1, testF1))
            print("(Epoch {}) BEST DEV F1: {:.4f} BEST TEST F1: {:.4f}".format(e+1, best_dev, best_test))
            if(f1 > best_dev):
                best_dev = f1
                best_test = testF1
                patience = 3
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


def eval_emotion(model, dataloader):
    model.eval()
    preds = []
    golds = []
    with torch.no_grad():
        for dialogs, lens, _, _, emotions, _, _, _ in dataloader:
            logits = model(dialogs, lens)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.detach().cpu().numpy())
            golds.append(emotions.cpu().numpy())

    preds = np.concatenate(preds)
    golds = np.concatenate(golds)
    f1 = f1_score(golds, preds, average='weighted')
    # _, _, _, microF1 = get_metrics(pred, gold, verbose=False if constant.grid_search else True)
    return f1
