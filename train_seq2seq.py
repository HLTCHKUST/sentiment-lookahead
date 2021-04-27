import os
import math
import random
import operator
from functools import reduce

import numpy as np
from tqdm import tqdm
from nltk.util import everygrams
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils import constant, tile, text_input2bert_input, EmbeddingSim
from utils.bleu import moses_multi_bleu
from utils.utils import get_metrics, save_ckpt, load_ckpt, save_model, load_model, distinct_ngrams, get_sentiment, get_user_response
from models import BinaryClassifier, RNNDecoder, RNNEncoder, Seq2Seq


def train_seq2seq(model, dataloaders):
    train_dataloader, dev_dataloader, test_dataloader = dataloaders
    criterion = nn.CrossEntropyLoss(ignore_index=constant.pad_idx)
    if constant.optim == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=constant.lr)
    elif constant.optim == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=constant.lr)
    else:
        print("Optim is not defined")
        exit(1)

    start_epoch = 1
    if constant.restore:
        model, opt, start_epoch = load_ckpt(model, opt, constant.restore_path)

    if constant.USE_CUDA:
        model.cuda()
        if constant.embeddings_cpu:
            model.encoder.embedding.cpu()

    best_dev = 10000
    best_path = ''
    patience = 3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'min', factor=0.5, patience=0, min_lr=1e-6)

    try:
        for e in range(start_epoch, constant.epochs):
            model.train()
            loss_log = []
            ppl_log = []

            if constant.grid_search:
                pbar = enumerate(train_dataloader)
            else:
                pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            
            for b, (dialogs, lens, targets, _, _, _, _, _, _) in pbar:
                if len(train_dataloader) % (b+1) == 10:
                    torch.cuda.empty_cache()
                opt.zero_grad()
                try:
                    batch_size, max_target_len = targets.shape
                    probs = model(dialogs, lens, targets)

                    # Masked CEL trick: Reshape probs to (B*L, V) and targets to (B*L,) and ignore pad idx
                    probs = probs.transpose(0, 1).contiguous().view(batch_size*max_target_len, -1)
                    targets = targets.contiguous().view(batch_size*max_target_len)
                    loss = criterion(probs, targets)
                    # if constant.embeddings_cpu and constant.USE_CUDA:
                    #     targets = targets.cuda()
                    #     target_lens = target_lens.cuda()
                    # loss = masked_cross_entropy(probs.transpose(0, 1).contiguous(), targets.contiguous(), target_lens)
                    loss.backward()
                    opt.step()

                    ## logging
                    loss_log.append(loss.item())
                    ppl_log.append(math.exp(loss_log[-1]))
                    if not constant.grid_search:
                        pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} TRAIN PPL:{:.1f}".format(e, np.mean(loss_log), np.mean(ppl_log)))
                except RuntimeError as err:
                    if 'out of memory' in str(err):
                        print('| WARNING: ran out of memory, skipping batch')
                        torch.cuda.empty_cache()
                    else:
                        raise err

            ## LOG
            dev_loss, dev_ppl = eval_seq2seq(model, dev_dataloader, bleu=False)
            
            print("(Epoch {}) DEV LOSS: {:.4f}, DEV PPL: {:.1f}".format(e, dev_loss, dev_ppl))

            scheduler.step(dev_loss)
            if(dev_loss < best_dev):
                best_dev = dev_loss
                # save best model
                path = 'trained/data-{}.task-seq2seq.lr-{}.emb-{}.D-{}.H-{}.attn-{}.bi-{}.parse-{}.loss-{}' # lr.embedding.D.H.attn.bi.parse.metric
                path = path.format(constant.data, constant.lr, constant.embedding, constant.D, constant.H, constant.attn, constant.bi, constant.parse, best_dev)
                if constant.topk:
                    path += '.topk-{}.tau-{}'.format(constant.topk_size, constant.tau)
                if constant.grid_search:
                    path += '.grid'
                best_path = save_model(model, 'loss', best_dev, path)
                patience = 3
            else:
                patience -= 1
            if patience == 0: break
            if best_dev == 0.0: break

    except KeyboardInterrupt:
        if not constant.grid_search:
            print("KEYBOARD INTERRUPT: Save CKPT and Eval")
            save = True if input('Save ckpt? (y/n)\t') in ['y', 'Y', 'yes', 'Yes'] else False
            if save:
                save_path = save_ckpt(model, opt, e)
                print("Saved CKPT path: ", save_path)
            # ask if eval
            do_eval = True if input('Proceed with eval? (y/n)\t') in ['y', 'Y', 'yes', 'Yes'] else False
            if do_eval:
                dev_loss, dev_ppl, dev_bleu, dev_bleus = eval_seq2seq(model, dev_dataloader, bleu=True, beam=constant.beam)
                print("DEV LOSS: {:.4f}, DEV PPL: {:.1f}, DEV BLEU: {:.4f}".format(dev_loss, dev_ppl, dev_bleu))
                print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
        exit(1)

    # load and report best model on test
    torch.cuda.empty_cache()
    model = load_model(model, best_path)
    if constant.USE_CUDA:
        model.cuda()

    # train_loss, train_ppl, train_bleu, train_bleus = eval_seq2seq(model, train_dataloader, bleu=True, beam=constant.beam)
    dev_loss, dev_ppl, dev_bleu, dev_bleus = eval_seq2seq(model, dev_dataloader, bleu=True, beam=constant.beam)
    test_loss, test_ppl, test_bleu, test_bleus = eval_seq2seq(model, test_dataloader, bleu=True, beam=constant.beam)

    # print("BEST TRAIN LOSS: {:.4f}, TRAIN PPL: {:.1f}, TRAIN BLEU: {:.4f}".format(train_loss, train_ppl, train_bleu))
    # print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(train_bleus[0], train_bleus[1], train_bleus[2], train_bleus[3]))

    print("BEST DEV LOSS: {:.4f}, DEV PPL: {:.1f}, DEV BLEU: {:.4f}".format(dev_loss, dev_ppl, dev_bleu))
    print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))

    print("BEST TEST LOSS: {:.4f}, TEST PPL: {:.1f}, TEST BLEU: {:.4f}".format(test_loss, test_ppl, test_bleu))
    print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))


def eval_seq2seq(model, dataloader, bleu=False, beam=False, raise_oom=False, test=False, save=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=constant.pad_idx)
    loss_log = []
    ppl_log = []
    vocab = dataloader.dataset.lang
    ctx = []
    ref = []
    g_hyps = []
    b_hyps = []
    bow_sims = []

    # automated metrics
    if test and bleu:
        embedding_metrics = EmbeddingSim(dataloader.dataset.fasttext)
        # define and load sentiment clf
        sentiment_clf = BinaryClassifier(encoder=BertModel.from_pretrained('bert-base-cased'), enc_type='bert', H=768)
        sentiment_clf = load_model(sentiment_clf, constant.sentiment_clf)

        # define and load user model
        encoder = RNNEncoder(V=len(dataloader.dataset.lang), D=constant.D, H=constant.H, L=1, embedding=None)
        decoder = RNNDecoder(V=len(dataloader.dataset.lang), D=constant.D, H=constant.H, L=1, embedding=None)
        user_model = Seq2Seq(encoder=encoder, decoder=decoder, vocab=dataloader.dataset.lang)
        user_model = load_model(user_model, constant.user_model)
        user_model.eval()

        if constant.USE_CUDA:
            sentiment_clf.cuda()
            user_model.cuda()

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        ref_lens = []
        gen_lens = []
        ref_sentiments = []
        gen_sentiments = []
        sentiment_agreement = []
        ref_improvement = []
        gen_improvement = []
        # distinct_ngrams = {
        #     'ref': set(),
        #     'gen': set()
        # }
        # total_ngrams = {
        #     'ref': 0,
        #     'gen': 0
        # }

    with torch.no_grad():
        try:
            for dialogs, lens, targets, unsort, _, _, _, _, _ in dataloader:
                logits = model(dialogs, lens, targets)
                
                if bleu:
                    # Calculate BLEU
                    probs, sents = model(dialogs, lens, targets, test=True)
                    # corrects: B x T
                    r = [" ".join([vocab.index2word[x_t] for x_t in iter(lambda x=iter(gens): next(x), constant.eou_idx)]) for gens in targets[unsort].cpu().data.numpy()]
                    c = [" ".join([vocab.index2word[x_t] for x_t in iter(lambda x=iter(gens): next(x), constant.pad_idx)]) for gens in dialogs[unsort].cpu().data.numpy()]
                    ref += r
                    ctx += c
                    
                    if test:
                        # calculate sentiment agreement
                        ref_sentiment = get_sentiment(sentiment_clf, r, tokenizer).squeeze() > 0.5
                        gen_sentiment = get_sentiment(sentiment_clf, np.array(sents)[unsort].tolist(), tokenizer).squeeze() > 0.5
                        sentiment_agreement += (ref_sentiment == gen_sentiment).cpu().numpy().tolist()
                        ref_sentiments += ref_sentiment.cpu().numpy().tolist()
                        gen_sentiments += gen_sentiment.cpu().numpy().tolist()

                        # calculate sentiment improvement
                        refs = [context + ' ' + sent for context, sent in zip(c, r)]
                        gens = [context + ' ' + sent for context, sent in zip(c, np.array(sents)[unsort].tolist())]

                        ref_simulation = get_user_response(user_model, targets, refs, model.vocab)
                        gen_simulation = get_user_response(user_model, targets, gens, model.vocab)

                        ctx_sentiment = get_sentiment(sentiment_clf, c, tokenizer).squeeze()
                        user_ref_sentiments = get_sentiment(sentiment_clf, ref_simulation, tokenizer).squeeze()
                        user_gen_sentiments = get_sentiment(sentiment_clf, gen_simulation, tokenizer).squeeze()
                        
                        ref_improvement += (user_ref_sentiments - ctx_sentiment).cpu().numpy().tolist()
                        gen_improvement += (user_gen_sentiments - ctx_sentiment).cpu().numpy().tolist()

                        # average generation lengths
                        ref_lens += [len(t.split()) for t in r]
                        gen_lens += [len(s.split()) for s in sents]

                        # calculate BoW embedding similarity
                        seqs = np.array([vocab.transform_one(sent) for sent in sents])
                        lens = [len(seq) for seq in seqs]
                        sort = np.argsort(lens)[::-1].tolist()
                        unsort = np.argsort(sort).tolist()
                        seqs = seqs[sort]
                        lens = np.array(lens)[sort].tolist()
                        padded_gens = np.ones((len(seqs), lens[0])).astype(int)
                        for b in range(len(seqs)):
                            padded_gens[b, :lens[b]] = np.array(seqs[b])
                    
                        extrema, avg, greedy = embedding_metrics.sim_bow(
                            padded_gens, 
                            lens, 
                            targets.cpu().numpy()[sort], 
                            [len(t.split()) for t in r])
                        bow_sims.append((extrema, avg, greedy))


                    if beam:
                        g_hyps += model.greedy_search(probs, vocab)
                        b_hyps += model.beam_search(dialogs, lens, targets.shape[0], targets.shape[1], vocab)
                    else:
                        g_hyps += np.array(sents)[unsort].tolist()
                
                # Masked CEL trick: Reshape logits to (B*L, V) and targets to (B*L,) and ignore pad idx
                batch_size, max_target_len = targets.shape
                logits = logits.transpose(0, 1).contiguous().view(batch_size*max_target_len, -1)
                targets = targets.contiguous().view(batch_size*max_target_len)
                loss = criterion(logits, targets)
                # loss = masked_cross_entropy(logits.transpose(0, 1).contiguous(), targets.contiguous(), target_lens)
                loss_log.append(loss.item())
                ppl_log.append(math.exp(loss_log[-1]))
        except RuntimeError as e:
            if 'out of memory' in str(e) and not raise_oom:
                print('| WARNING: ran out of memory, retrying batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return eval_seq2seq(model, dataloader, bleu, raise_oom=True)
            else:
                raise e
        
    if not constant.grid_search:
        if save:
            if bleu and test:
                if not constant.topk:
                    fname = "samples/{}.greedy.txt".format(constant.test_path.split('/')[1])
                else:
                    fname = "samples/{}.topk.{:.4f}.txt".format(constant.test_path.split('/')[1], pearsonr(ref_sentiments, gen_sentiments)[0])
            else:
                fname = "samples/{}.greedy.txt".format(constant.test_path.split('/')[1])
            with open(fname, "w") as f:
                for i, (c, r, h) in enumerate(zip(ctx, ref, g_hyps)):
                    f.write("DIAL {}: {}\n".format(i, c))
                    f.write("GOLD: {}\n".format(r))
                    f.write("PRED: {}\n".format(h))
                    f.write("\n")
        else:
            count = 0
            if not beam:
                for c, r, h in zip(ctx, ref, g_hyps):
                    if count < 100:
                        print("DIAL: ", c)
                        print("GOLD: ", r)
                        print("PRED: ", h)
                        print()
                        count += 1
                    else: 
                        break
            else:
                for c, r, g, b in zip(ctx, ref, g_hyps, b_hyps):
                    if count < 100:
                        print("DIAL: ")
                        print(c)
                        print("GOLD: ")
                        print(r)
                        print("GRDY: ")
                        print(g)
                        print("BEAM: ")
                        print(b)
                        print()
                        count += 1
                    else: 
                        break
    
    if bleu:
        hyps = b_hyps if beam else g_hyps
        bleu_score, bleus = moses_multi_bleu(np.array(hyps), np.array(ref), lowercase=True)
        bow_sims = np.array(bow_sims)
        if test:
            return np.mean(loss_log), np.mean(ppl_log), bleu_score, bleus, np.mean(bleus), np.mean(ref_lens), np.mean(gen_lens), distinct_ngrams(ref), distinct_ngrams(g_hyps), pearsonr(ref_sentiments, gen_sentiments)[0], sum(sentiment_agreement) / len(sentiment_agreement), np.mean(ref_improvement), np.mean(gen_improvement), np.mean(bow_sims, axis=0)
        else:
            return np.mean(loss_log), np.mean(ppl_log), bleu_score, bleus
    else:
        return np.mean(loss_log), np.mean(ppl_log)
