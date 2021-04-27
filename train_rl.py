import os
import math
import random
import operator
import traceback
from functools import reduce

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from nltk.util import everygrams
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils import constant, tile, text_input2bert_input, EmbeddingSim
from utils.bleu import moses_multi_bleu
from utils.utils import get_metrics, save_ckpt, load_ckpt, save_model, load_model, distinct_ngrams, get_sentiment, get_user_response
from models import BinaryClassifier, RNNDecoder, RNNEncoder, Seq2Seq


def train_rl(model, dataloaders):
    train_dataloader, dev_dataloader, test_dataloader = dataloaders
    
    clf_criterion = nn.BCEWithLogitsLoss()
    mle_criterion = nn.CrossEntropyLoss(ignore_index=constant.pad_idx)
    baseline_criterion = nn.MSELoss()

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

    best_dev = 0
    best_path = ''
    patience = 3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'max', factor=0.5, patience=0, min_lr=1e-6)
    tau = constant.tau
    tau_min = 0.2
    tau_dec = 0.2
    pretrain_curiosity = constant.lambda_aux
    if constant.pretrain_curiosity:
        pretrain_curiosity = 0.0

    try:
        for e in range(start_epoch, constant.epochs):
            model.train()
            reward_log = []
            ori_reward_log = []
            aux_reward_log = [] # for sentiment agreement / curiosity
            inv_loss_log = [] # for curiosity
            f1_log = []

            # pretrain curiosity only for first epoch
            if e > start_epoch:
                pretrain_curiosity = constant.lambda_aux

            # temperature annealing
            if constant.use_tau_anneal and e > start_epoch and constant.tau > tau_min:
                constant.tau -= tau_dec

            if constant.grid_search:
                pbar = enumerate(train_dataloader)
            else:
                pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            
            for b, (dialogs, lens, targets, _, _, sentiments, sentiments_b, _, _) in pbar:
                if len(train_dataloader) % (b+1) == 10:
                    torch.cuda.empty_cache()
                opt.zero_grad()
                try:
                    B, T = targets.shape

                    if constant.use_self_critical:
                        step_loss, dec_lens_var, R_g, R, greedy_sents, sampled_sents = model(dialogs, lens, targets)
                        # (R_s - R_g) * step_loss
                        rl_loss = torch.mean(torch.sum((R.detach() - R_g.detach()) * step_loss, dim=1) / dec_lens_var.float())
                    elif constant.use_arl:
                        step_loss, dec_lens_var, rs, R, arl, sampled_sents = model(dialogs, lens, targets)
                        rs = rs.transpose(0, 1).contiguous()
                        
                        rl_loss = (R.detach() - rs.detach()) * step_loss
                        rl_loss = torch.mean(torch.sum(rl_loss * arl, dim=1) / dec_lens_var.float())
                    else:
                        # probs: (B, T, V), xs: (B, T), R: (B, 1), rs: (B, T)
                        if constant.use_sentiment and constant.aux_reward_model != '':
                            step_loss, dec_lens_var, rs, R_l, R_s, sampled_sents, clf_logits = model(dialogs, lens, targets, sentiments=sentiments)
                            R = constant.lambda_aux * R_l + R_s
                            clf_loss = clf_criterion(clf_logits, sentiments_b)
                            preds = torch.sigmoid(clf_logits.squeeze()) > 0.5
                            f1 = f1_score(sentiments_b.cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
                            f1_log.append(f1)
                        elif constant.use_sentiment and constant.use_sentiment_agreement:
                            step_loss, dec_lens_var, rs, R, sampled_sents, clf_logits = model(dialogs, lens, targets, sentiments=sentiments)
                            clf_loss = clf_criterion(clf_logits, sentiments_b)
                            preds = torch.sigmoid(clf_logits.squeeze()) > 0.5
                            f1 = f1_score(sentiments_b.cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
                            f1_log.append(f1)
                        elif constant.use_sentiment_agreement:
                            step_loss, dec_lens_var, rs, R, sampled_sents = model(dialogs, lens, targets, sentiments=sentiments)
                        elif constant.use_sentiment:
                            step_loss, dec_lens_var, rs, R, sampled_sents, clf_logits = model(dialogs, lens, targets, sentiments=sentiments)
                            clf_loss = clf_criterion(clf_logits, sentiments_b)
                            preds = torch.sigmoid(clf_logits.squeeze()) > 0.5
                            f1 = f1_score(sentiments_b.cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
                            f1_log.append(f1)
                        elif constant.use_curiosity:
                            step_loss, dec_lens_var, rs, R, R_i, L_i, sampled_sents = model(dialogs, lens, targets)
                            rs = rs.transpose(0, 1).contiguous()
                            R_i = R_i.transpose(0, 1).contiguous()
                            baseline_target = R.detach() * R_i.detach()
                            rl_loss = torch.mean(torch.sum((R.detach() * R_i.detach() - rs.detach()) * step_loss, dim=1) / dec_lens_var.float())
                            R_i = torch.mean(torch.sum(R_i, dim=1) / dec_lens_var.float())
                        else:
                            step_loss, dec_lens_var, rs, R, sampled_sents = model(dialogs, lens, targets)
                        
                        if not constant.use_curiosity:
                            # probs = probs.transpose(0, 1).cntiguous()
                            # xs = xs.transpose(0, 1).contiguous()
                            # # (B, T, V) => (B, T) => (B,)
                            # probs = torch.gather(probs, dim=2, index=xs.unsqueeze(2)).squeeze()
                            # probs = -torch.log(probs)
                            rs = rs.transpose(0, 1).contiguous()
                            rl_loss = torch.mean(torch.sum((R.detach() - rs.detach()) * step_loss, dim=1) / dec_lens_var.float())

                    if constant.use_hybrid:
                        probs, _ = model(dialogs, lens, targets, use_mle=True)
                        mle_loss = mle_criterion(probs.transpose(0, 1).contiguous().view(B*T, -1), targets.contiguous().view(B*T))
                        loss = constant.lambda_mle * rl_loss + (1 - constant.lambda_mle) * mle_loss
                    elif constant.use_arl:
                        probs, _ = model(dialogs, lens, targets, use_mle=True)
                        arl_c = torch.ones(arl.size()).to(arl.device) - arl
                        mle_criterion.reduction = 'none'
                        mle_loss = mle_criterion(probs.transpose(0, 1).contiguous().view(B*T, -1), targets.contiguous().view(B*T))
                        mle_loss = torch.mean(torch.sum(mle_loss * arl_c, dim=1))
                        loss = rl_loss + mle_loss
                    else:
                        loss = rl_loss

                    if constant.use_sentiment:
                        loss = constant.lambda_emo * clf_loss + (1 - constant.lambda_emo) * loss
                    
                    if constant.use_curiosity:
                        loss = pretrain_curiosity * loss + (1 - constant.beta) * L_i + constant.beta * R_i

                    loss.backward()
                    opt.step()
                    
                    if constant.use_baseline:
                        if constant.use_curiosity:
                            baseline_loss = baseline_criterion(rs, baseline_target)
                        else:
                            # rs (32, T) <==> R (32, 1)
                            baseline_loss = baseline_criterion(rs, tile(R, T, dim=1))
                        baseline_loss.backward()
                        opt.step()

                    ## logging
                    reward_log.append(torch.mean(R).item())
                    if constant.use_sentiment and constant.aux_reward_model != '':
                        ori_reward_log.append(torch.mean(R_l).item())
                        aux_reward_log.append(torch.mean(R_s).item())

                    if constant.use_curiosity:
                        aux_reward_log.append(torch.mean(R_i).item())
                        inv_loss_log.append(L_i.item())

                    if not constant.grid_search:
                        if constant.use_sentiment:
                            if constant.aux_reward_model != '':
                                pbar.set_description("(Epoch {}) TRAIN R: {:.3f} R_l: {:.3f} R_s: {:.3f} F1: {:.3f}".format(e, np.mean(reward_log), np.mean(ori_reward_log), np.mean(aux_reward_log), np.mean(f1_log)))
                            else:
                                pbar.set_description("(Epoch {}) TRAIN REWARD: {:.4f} TRAIN F1: {:.4f}".format(e, np.mean(reward_log), np.mean(f1_log)))
                        elif constant.use_curiosity:
                            pbar.set_description("(Epoch {}) TRAIN R: {:.3f} R_i: {:.3f} L_i: {:.3f}".format(e, np.mean(reward_log), np.mean(aux_reward_log), np.mean(inv_loss_log)))
                        else:
                            pbar.set_description("(Epoch {}) TRAIN REWARD: {:.4f}".format(e, np.mean(reward_log)))

                    if b % 100 == 0 and b > 0:
                        # if not constant.use_self_critical:
                        #     _, greedy_sents = model(dialogs, lens, targets, test=True, use_mle=True)
                        corrects = [" ".join([train_dataloader.dataset.lang.index2word[x_t] for x_t in iter(lambda x=iter(gens): next(x), constant.eou_idx)]) for gens in targets.cpu().data.numpy()] 
                        contexts = [" ".join([train_dataloader.dataset.lang.index2word[x_t] for x_t in iter(lambda x=iter(gens): next(x), constant.pad_idx)]) for gens in dialogs.cpu().data.numpy()] 
                        for d, c, s, r in zip(contexts, corrects, sampled_sents, R.detach().cpu().numpy()):
                            print('reward: ', r)
                            print('dialog: ', d)
                            print('sample: ', s)
                            print('golden: ', c)
                            print()
                except RuntimeError as err:
                    if 'out of memory' in str(err):
                        print('| WARNING: ran out of memory, skipping batch')
                        torch.cuda.empty_cache()
                    else:
                        print(err)
                        traceback.print_exc()
                        raise err

            ## LOG
            if constant.use_sentiment and not constant.use_sentiment_agreement:
                dev_reward, dev_f1 = eval_rl(model, dev_dataloader, bleu=False)
                print("(Epoch {}) DEV REWARD: {:.4f}".format(e, dev_reward))
            elif constant.use_curiosity:
                dev_reward, dev_Ri, dev_Li = eval_rl(model, dev_dataloader, bleu=False)
                print("(Epoch {}) DEV REWARD: {:.3f} R_i: {:.3f} L_i: {:.3f}".format(e, dev_reward, dev_Ri, dev_Li))
            else:
                dev_reward = eval_rl(model, dev_dataloader, bleu=False)
                print("(Epoch {}) DEV REWARD: {:.4f}".format(e, dev_reward))

            scheduler.step(dev_reward)
            if(dev_reward > best_dev):
                best_dev = dev_reward
                # save best model
                path = 'trained/data-{}.task-rlseq.lr-{}.tau-{}.lambda-{}.reward-{}.{}'
                path = path.format(constant.data, constant.lr, tau, constant.lambda_mle, best_dev, constant.reward_model.split('/')[1])
                if constant.use_curiosity:
                    path += '.curiosity'
                if constant.aux_reward_model != '':
                    path += '.' + constant.aux_reward_model.split('/')[1]
                    path += '.lambda_aux-{}'.format(constant.lambda_aux)
                if constant.use_tau_anneal:
                    path += '.tau_anneal'
                if constant.use_self_critical:
                    path += '.self_critical'
                if constant.use_current:
                    path += '.current'
                if constant.use_sentiment:
                    path += '.sentiment'
                if constant.use_sentiment_agreement:
                    path += '.agreement'
                if constant.use_context:
                    path += '.context'
                if constant.topk:
                    path += '.topk-{}'.format(constant.topk_size)
                if constant.use_arl:
                    path += '.arl'
                if constant.grid_search:
                    path += '.grid'
                best_path = save_model(model, 'reward', best_dev, path)
                patience = 3
            else:
                patience -= 1
            if patience == 0: break
            if constant.aux_reward_model == '' and best_dev == 0.0: break

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
                if constant.use_sentiment:
                    if constant.aux_reward_model != '':
                        dev_rewards, dev_f1, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
                        print("DEV R: {:.3f} R_l: {:.3f} R_s: {:.3f} DEV F1: {:.3f} DEV B: {:.3f}".format(dev_rewards[0], dev_rewards[1], dev_rewards[2], dev_f1, dev_bleu))
                    else:
                        dev_reward, dev_f1, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
                        print("DEV REWARD: {:.4f}, DEV F1: {:.4f}, DEV BLEU: {:.4f}".format(dev_reward, dev_f1, dev_bleu))
                elif constant.use_curiosity:
                    dev_reward, dev_Ri, dev_Li, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
                    print("BEST DEV REWARD: {:.4f} R_i: {:.3f} L_i: {:.3f} BLEU: {:.4f}".format(dev_reward, dev_Ri, dev_Li, dev_bleu))
                else:
                    dev_reward, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
                    print("DEV REWARD: {:.4f}, DEV BLEU: {:.4f}".format(dev_reward, dev_bleu))
                print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
        exit(1)

    # load and report best model on test
    torch.cuda.empty_cache()
    model = load_model(model, best_path)
    if constant.USE_CUDA:
        model.cuda()

    if constant.use_sentiment and not constant.use_sentiment_agreement:
        if constant.aux_reward_model != '':
            dev_rewards, dev_f1, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
            test_rewards, test_f1, test_bleu, test_bleus = eval_rl(model, test_dataloader, bleu=True)
            print("DEV R: {:.3f} R_l: {:.3f} R_s: {:.3f} DEV F1: {:.3f} DEV B: {:.3f}".format(dev_rewards[0], dev_rewards[1], dev_rewards[2], dev_f1, dev_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
            print("TEST R: {:.3f} R_l: {:.3f} R_s: {:.3f} TEST F1: {:.3f} TEST B: {:.3f}".format(test_rewards[0], test_rewards[1], test_rewards[2], test_f1, test_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
        else:
            dev_reward, dev_f1, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
            test_reward, test_f1, test_bleu, test_bleus = eval_rl(model, test_dataloader, bleu=True)
            print("DEV REWARD: {:.4f}, DEV F1: {:.4f}, DEV BLEU: {:.4f}".format(dev_reward, dev_f1, dev_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
            print("TEST REWARD: {:.4f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}".format(test_reward, test_f1, test_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
    elif constant.use_curiosity:
        dev_reward, dev_Ri, dev_Li, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
        test_reward, test_Ri, test_Li, test_bleu, test_bleus = eval_rl(model, test_dataloader, bleu=True)
        print("BEST DEV REWARD: {:.4f} R_i: {:.3f} L_i: {:.3f} BLEU: {:.4f}".format(dev_reward, dev_Ri, dev_Li, dev_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
        print("BEST TEST REWARD: {:.4f} R_i: {:.3f} L_i: {:.3f} BLEU: {:.4f}".format(test_reward, test_Ri, test_Li, test_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
    else:
        dev_reward, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
        test_reward, test_bleu, test_bleus = eval_rl(model, test_dataloader, bleu=True)
        print("BEST DEV REWARD: {:.4f}, BLEU: {:.4f}".format(dev_reward, dev_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
        print("BEST TEST REWARD: {:.4f}, BLEU: {:.4f}".format(test_reward, test_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))


def eval_rl(model, dataloader, bleu=False, raise_oom=False, save=False, test=False):
    model.eval()
    preds = []
    golds = []
    reward_log = []
    ori_reward_log = []
    aux_reward_log = []
    inv_loss_log = []
    vocab = dataloader.dataset.lang
    ctx = []
    ref = []
    g_hyps = []
    bow_sims = []
    # mle_criterion = nn.CrossEntropyLoss(ignore_index=constant.pad_idx)

    # automated metrics
    if test and bleu:
        tokenizer = model.reward_tokenizer
        embedding_metrics = EmbeddingSim(dataloader.dataset.fasttext)

        # define and load sentiment clf
        if constant.reward_model == constant.sentiment_clf:
            sentiment_clf = model.reward
        else:
            sentiment_clf = BinaryClassifier(encoder=BertModel.from_pretrained('bert-base-cased'), enc_type='bert', H=768)
            sentiment_clf = load_model(sentiment_clf, constant.sentiment_clf)

        if constant.use_user:
            user_model = model.user_model
        else:
            # define and load user model
            encoder = RNNEncoder(V=len(dataloader.dataset.lang), D=constant.D, H=constant.H, L=1, embedding=None)
            decoder = RNNDecoder(V=len(dataloader.dataset.lang), D=constant.D, H=constant.H, L=1, embedding=None)
            user_model = Seq2Seq(encoder=encoder, decoder=decoder, vocab=dataloader.dataset.lang)
            user_model = load_model(user_model, constant.user_model)
            user_model.eval()

        if constant.USE_CUDA:
            sentiment_clf.cuda()
            user_model.cuda()

        ref_lens = []
        gen_lens = []
        ref_sentiments = []
        gen_sentiments = []
        ref_improvement = []
        gen_improvement = []
        sentiment_agreement = []

    with torch.no_grad():
        try:
            for dialogs, lens, targets, unsort, _, sentiments, sentiments_b, _, _ in dataloader:
                if constant.use_sentiment:
                    if constant.aux_reward_model != '':
                        _, _, _, R_l, R_s, _, clf_logits = model(dialogs, lens, targets, sentiments=sentiments)
                        R = constant.lambda_aux * R_l + R_s
                        ori_reward_log.append(torch.mean(R_l).item())
                        aux_reward_log.append(torch.mean(R_s).item())
                    else:
                        _, _, _, R, _, clf_logits = model(dialogs, lens, targets, sentiments=sentiments)
                    pred = torch.sigmoid(clf_logits.squeeze()) > 0.5
                    preds.append(pred.detach().cpu().numpy())
                    golds.append(sentiments_b.cpu().numpy())
                elif constant.use_sentiment_agreement:
                    _, _, _, R, _ = model(dialogs, lens, targets, sentiments=sentiments)
                elif constant.use_curiosity:
                    _, dec_lens_var, _, R, R_i, L_i, _ = model(dialogs, lens, targets)
                    R_i = torch.mean(torch.sum(R_i.transpose(0, 1).contiguous(), dim=1) / dec_lens_var.float())
                    aux_reward_log.append(torch.mean(R_i).item())
                    inv_loss_log.append(L_i.item())
                else:
                    _, _, _, R, _ = model(dialogs, lens, targets, sentiments=sentiments, test=True)
                reward_log.append(torch.mean(R).item())

                if bleu:
                    # Calculate BLEU
                    _, sents = model(dialogs, lens, targets, test=True, use_mle=True)

                    g_hyps += np.array(sents)[unsort].tolist()
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

                
        except RuntimeError as e:
            if 'out of memory' in str(e) and not raise_oom:
                print('| WARNING: ran out of memory, retrying batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return eval_rl(model, dataloader, bleu, raise_oom=True)
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
            for c, r, h in zip(ctx, ref, g_hyps):
                if count < 100:
                    print("DIAL: ", c)
                    print("GOLD: ", r)
                    print("GRDY: ", h)
                    print()
                    count += 1
                else: 
                    break
    
    if bleu:
        bleu_score, bleus = moses_multi_bleu(np.array(g_hyps), np.array(ref), lowercase=True)
        if test:
            bow_sims = np.array(bow_sims)
            if constant.use_sentiment and constant.aux_reward_model != '':
                return [np.mean(reward_log), np.mean(ori_reward_log), np.mean(aux_reward_log)], bleu_score, bleus
            elif constant.use_sentiment:
                preds = np.hstack(np.array(preds))
                golds = np.concatenate(golds)
                f1 = f1_score(preds, golds, average='weighted')
                return np.mean(reward_log), f1, bleu_score, bleus, np.mean(bleus), np.mean(ref_lens), np.mean(gen_lens), distinct_ngrams(ref), distinct_ngrams(g_hyps), pearsonr(ref_sentiments, gen_sentiments)[0], sum(sentiment_agreement) / len(sentiment_agreement), np.mean(ref_improvement), np.mean(gen_improvement), np.mean(bow_sims, axis=0)
            elif constant.use_curiosity:
                return np.mean(reward_log), np.mean(aux_reward_log), np.mean(inv_loss_log), bleu_score, bleus
            else:
                return np.mean(reward_log), bleu_score, bleus, np.mean(bleus), np.mean(ref_lens), np.mean(gen_lens), distinct_ngrams(ref), distinct_ngrams(g_hyps), pearsonr(ref_sentiments, gen_sentiments)[0], sum(sentiment_agreement) / len(sentiment_agreement), np.mean(ref_improvement), np.mean(gen_improvement), np.mean(bow_sims, axis=0)
        elif constant.use_curiosity:
            return np.mean(reward_log), np.mean(aux_reward_log), np.mean(inv_loss_log), bleu_score, bleus
        elif constant.use_sentiment:
            if constant.use_sentiment_agreement:
                return np.mean(reward_log), bleu_score, bleus
            preds = np.hstack(np.array(preds))
            golds = np.concatenate(golds)
            f1 = f1_score(preds, golds, average='weighted')
            if constant.aux_reward_model != '':
                return [np.mean(reward_log), np.mean(ori_reward_log), np.mean(aux_reward_log)], f1, bleu_score, bleus
            else:
                return np.mean(reward_log), f1, bleu_score, bleus
        else:
            return np.mean(reward_log), bleu_score, bleus
    else:
        if test:
            if constant.use_curiosity:
                return np.mean(reward_log), np.mean(aux_reward_log), np.mean(inv_loss_log)
            return np.mean(reward_log)
        elif constant.use_curiosity:
            return np.mean(reward_log), np.mean(aux_reward_log), np.mean(inv_loss_log)
        elif constant.use_sentiment:
            if constant.use_sentiment_agreement:
                return np.mean(reward_log)
            preds = np.hstack(np.array(preds))
            golds = np.concatenate(golds)
            f1 = f1_score(preds, golds, average='weighted')
            return np.mean(reward_log), f1
        else:
            return np.mean(reward_log)
