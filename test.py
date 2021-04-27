import os
import math
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import constant, masked_cross_entropy
from utils.bleu import moses_multi_bleu
from utils.utils import get_metrics, load_ckpt, load_model

from train_rl import train_rl, eval_rl
from train_emotion import train_emotion, eval_emotion
from train_seq2seq import train_seq2seq, eval_seq2seq
from train_multitask import train_multitask, eval_multitask


def test(model, dataloaders, test_path=''):
    # load and report best model on test
    _, dev_dataloader, test_dataloader = dataloaders
    if test_path != '':
        model = load_model(model, test_path)
    if(constant.USE_CUDA): model.cuda()
    
    if constant.task == 'emotion':
        pass
    elif constant.task == 'seq2seq':
        dev_loss, dev_ppl, dev_bleu, dev_bleus, dev_avg_bleu, dev_ref_lens, dev_gen_lens, dev_ref_ngrams, dev_gen_ngrams, dev_sentiment_correlation, dev_sentiment_accuracy, dev_ref_improvement, dev_gen_improvement, dev_bow_sims = eval_seq2seq(model, dev_dataloader, bleu=True, beam=constant.beam, test=True)
        test_loss, test_ppl, test_bleu, test_bleus, test_avg_bleu, test_ref_lens, test_gen_lens, test_ref_ngrams, test_gen_ngrams, test_sentiment_correlation, test_sentiment_accuracy, test_ref_improvement, test_gen_improvement, test_bow_sims = eval_seq2seq(model, test_dataloader, bleu=True, beam=constant.beam, test=True, save=True)

        print("BEST DEV LOSS: {:.4f}, DEV PPL: {:.1f}, DEV BLEU: {:.4f}, AVG BLEU: {:.4f}".format(dev_loss, dev_ppl, dev_bleu, dev_avg_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
        print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(dev_ref_ngrams[0], dev_ref_ngrams[1], dev_ref_ngrams[2]))
        print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(dev_gen_ngrams[0], dev_gen_ngrams[1], dev_gen_ngrams[2]))
        print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(dev_bow_sims[0], dev_bow_sims[1], dev_bow_sims[2]))
        print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(dev_ref_lens, dev_gen_lens, dev_sentiment_accuracy, dev_sentiment_correlation, dev_ref_improvement, dev_gen_improvement))

        print("BEST TEST LOSS: {:.4f}, TEST PPL: {:.1f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_loss, test_ppl, test_bleu, test_avg_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
        print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
        print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
        print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
        print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

        with open("samples/{}_summary.txt".format(constant.test_path.split('/')[1]), "w") as f:
            f.write("BEST TEST LOSS: {:.4f}, TEST PPL: {:.1f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_loss, test_ppl, test_bleu, test_avg_bleu))
            f.write("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
            f.write("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
            f.write("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
            f.write("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
            f.write("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

    elif constant.task == 'rlseq':
        if constant.use_sentiment and not constant.use_sentiment_agreement:
            if constant.aux_reward_model != '':
                dev_rewards, dev_f1, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
                test_rewards, test_f1, test_bleu, test_bleus = eval_rl(model, test_dataloader, bleu=True, save=True)
                print("DEV R: {:.3f} R_l: {:.3f} R_s: {:.3f} DEV F1: {:.3f} DEV B: {:.3f}".format(dev_rewards[0], dev_rewards[1], dev_rewards[2], dev_f1, dev_bleu))
                print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
                print("TEST R: {:.3f} R_l: {:.3f} R_s: {:.3f} TEST F1: {:.3f} TEST B: {:.3f}".format(test_rewards[0], test_rewards[1], test_rewards[2], test_f1, test_bleu))
                print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
            else:
                dev_reward, dev_f1, dev_bleu, dev_bleus, dev_avg_bleu, dev_ref_lens, dev_gen_lens, dev_ref_ngrams, dev_gen_ngrams, dev_sentiment_correlation, dev_sentiment_accuracy, dev_ref_improvement, dev_gen_improvement, dev_bow_sims = eval_rl(model, dev_dataloader, bleu=True, test=True)
                test_reward, test_f1, test_bleu, test_bleus, test_avg_bleu, test_ref_lens, test_gen_lens, test_ref_ngrams, test_gen_ngrams, test_sentiment_correlation, test_sentiment_accuracy, test_ref_improvement, test_gen_improvement, test_bow_sims  = eval_rl(model, test_dataloader, bleu=True, test=True, save=True)
                print("BEST DEV REWARD: {:.4f}, DEV F1: {:.4f}, DEV BLEU: {:.4f}, AVG BLEU: {:.4f}".format(dev_reward, dev_f1, dev_bleu, dev_avg_bleu))
                print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
                print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(dev_ref_ngrams[0], dev_ref_ngrams[1], dev_ref_ngrams[2]))
                print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(dev_gen_ngrams[0], dev_gen_ngrams[1], dev_gen_ngrams[2]))
                print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(dev_bow_sims[0], dev_bow_sims[1], dev_bow_sims[2]))
                print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(dev_ref_lens, dev_gen_lens, dev_sentiment_accuracy, dev_sentiment_correlation, dev_ref_improvement, dev_gen_improvement))

                print("BEST TEST REWARD: {:.4f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_reward, test_f1, test_bleu, test_avg_bleu))
                print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
                print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
                print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
                print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
                print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

                with open("samples/{}_summary.txt".format(constant.test_path.split('/')[1]), "w") as f:
                    f.write("BEST TEST REWARD: {:.4f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_reward, test_f1, test_bleu, test_avg_bleu))
                    f.write("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
                    f.write("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
                    f.write("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
                    f.write("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
                    f.write("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

        elif constant.use_sentiment and constant.use_sentiment_agreement:
            dev_reward, dev_f1, dev_bleu, dev_bleus, dev_avg_bleu, dev_ref_lens, dev_gen_lens, dev_ref_ngrams, dev_gen_ngrams, dev_sentiment_correlation, dev_sentiment_accuracy, dev_ref_improvement, dev_gen_improvement, dev_bow_sims = eval_rl(model, dev_dataloader, bleu=True, test=True)
            test_reward, test_f1, test_bleu, test_bleus, test_avg_bleu, test_ref_lens, test_gen_lens, test_ref_ngrams, test_gen_ngrams, test_sentiment_correlation, test_sentiment_accuracy, test_ref_improvement, test_gen_improvement, test_bow_sims = eval_rl(model, test_dataloader, bleu=True, test=True, save=True)
            print("BEST DEV REWARD: {:.4f}, DEV F1: {:.4f}, DEV BLEU: {:.4f}, AVG BLEU: {:.4f}".format(dev_reward, dev_f1, dev_bleu, dev_avg_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
            print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(dev_ref_ngrams[0], dev_ref_ngrams[1], dev_ref_ngrams[2]))
            print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(dev_gen_ngrams[0], dev_gen_ngrams[1], dev_gen_ngrams[2]))
            print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(dev_bow_sims[0], dev_bow_sims[1], dev_bow_sims[2]))
            print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(dev_ref_lens, dev_gen_lens, dev_sentiment_accuracy, dev_sentiment_correlation, dev_ref_improvement, dev_gen_improvement))

            print("BEST TEST REWARD: {:.4f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_reward, test_f1, test_bleu, test_avg_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
            print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
            print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
            print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
            print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

            with open("samples/{}_summary.txt".format(constant.test_path.split('/')[1]), "w") as f:
                f.write("BEST TEST REWARD: {:.4f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_reward, test_f1, test_bleu, test_avg_bleu))
                f.write("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
                f.write("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
                f.write("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
                f.write("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
                f.write("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

        elif constant.use_curiosity:
            # np.mean(ref_lens), np.mean(gen_lens), len(distinct_ngrams['ref']) / total_ngrams['ref'], len(distinct_ngrams['gen']) / total_ngrams['gen'], sum(sentiment_agreement) / len(sentiment_agreement)
            dev_reward, dev_Ri, dev_Li, dev_bleu, dev_bleus = eval_rl(model, dev_dataloader, bleu=True)
            test_reward, test_Ri, test_Li, test_bleu, test_bleus = eval_rl(model, test_dataloader, bleu=True, save=True)
            print("BEST DEV REWARD: {:.4f} R_i: {:.3f} L_i: {:.3f} BLEU: {:.4f}".format(dev_reward, dev_Ri, dev_Li, dev_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))

            print("BEST TEST REWARD: {:.4f} R_i: {:.3f} L_i: {:.3f} BLEU: {:.4f}".format(test_reward, test_Ri, test_Li, test_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
      
        else:
            dev_reward, dev_bleu, dev_bleus, dev_avg_bleu, dev_ref_lens, dev_gen_lens, dev_ref_ngrams, dev_gen_ngrams, dev_sentiment_correlation, dev_sentiment_accuracy, dev_ref_improvement, dev_gen_improvement, dev_bow_sims = eval_rl(model, dev_dataloader, bleu=True, test=True)
            test_reward, test_bleu, test_bleus, test_avg_bleu, test_ref_lens, test_gen_lens, test_ref_ngrams, test_gen_ngrams, test_sentiment_correlation, test_sentiment_accuracy, test_ref_improvement, test_gen_improvement, test_bow_sims = eval_rl(model, test_dataloader, bleu=True, test=True, save=True)
            print("BEST DEV REWARD: {:.4f}, DEV BLEU: {:.4f}, AVG BLEU: {:.4f}".format(dev_reward, dev_bleu, dev_avg_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
            print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(dev_ref_ngrams[0], dev_ref_ngrams[1], dev_ref_ngrams[2]))
            print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(dev_gen_ngrams[0], dev_gen_ngrams[1], dev_gen_ngrams[2]))
            print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(dev_bow_sims[0], dev_bow_sims[1], dev_bow_sims[2]))
            print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(dev_ref_lens, dev_gen_lens, dev_sentiment_accuracy, dev_sentiment_correlation, dev_ref_improvement, dev_gen_improvement))

            print("BEST TEST REWARD: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_reward, test_bleu, test_avg_bleu))
            print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
            print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
            print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
            print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
            print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

            with open("samples/{}_summary.txt".format(constant.test_path.split('/')[1]), "w") as f:
                f.write("BEST TEST REWARD: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_reward, test_bleu, test_avg_bleu))
                f.write("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
                f.write("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
                f.write("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
                f.write("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
                f.write("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

    elif constant.task == 'multiseq':
        (dev_loss, dev_ppl, dev_bleu, dev_bleus, dev_avg_bleu, dev_ref_lens, dev_gen_lens, dev_ref_ngrams, dev_gen_ngrams, dev_sentiment_correlation, dev_sentiment_accuracy, dev_ref_improvement, dev_gen_improvement, dev_bow_sims), (dev_f1) = eval_multitask(model, dev_dataloader, bleu=True, beam=constant.beam, test=True)
        (test_loss, test_ppl, test_bleu, test_bleus, test_avg_bleu, test_ref_lens, test_gen_lens, test_ref_ngrams, test_gen_ngrams, test_sentiment_correlation, test_sentiment_accuracy, test_ref_improvement, test_gen_improvement, test_bow_sims), (test_f1) = eval_multitask(model, test_dataloader, bleu=True, beam=constant.beam, test=True, save=True)

        print("BEST DEV LOSS: {:.4f}, DEV PPL: {:.1f}, DEV F1: {:.4f}, DEV BLEU: {:.4f}, AVG BLEU: {:.4f}".format(dev_loss, dev_ppl, dev_f1, dev_bleu, dev_avg_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(dev_bleus[0], dev_bleus[1], dev_bleus[2], dev_bleus[3]))
        print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(dev_ref_ngrams[0], dev_ref_ngrams[1], dev_ref_ngrams[2]))
        print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(dev_gen_ngrams[0], dev_gen_ngrams[1], dev_gen_ngrams[2]))
        print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(dev_bow_sims[0], dev_bow_sims[1], dev_bow_sims[2]))
        print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(dev_ref_lens, dev_gen_lens, dev_sentiment_accuracy, dev_sentiment_correlation, dev_ref_improvement, dev_gen_improvement))

        print("BEST TEST LOSS: {:.4f}, TEST PPL: {:.1f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_loss, test_ppl, test_f1, test_bleu, test_avg_bleu))
        print("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
        print("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
        print("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
        print("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
        print("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))

        with open("samples/{}_summary.txt".format(constant.test_path.split('/')[1]), "w") as f:
            f.write("BEST TEST LOSS: {:.4f}, TEST PPL: {:.1f}, TEST F1: {:.4f}, TEST BLEU: {:.4f}, AVG BLEU: {:.4f}".format(test_loss, test_ppl, test_f1, test_bleu, test_avg_bleu))
            f.write("BLEU 1: {:.4f}, BLEU 2: {:.4f}, BLEU 3: {:.4f}, BLEU 4: {:.4f}".format(test_bleus[0], test_bleus[1], test_bleus[2], test_bleus[3]))
            f.write("REF 1-GRAMS: {:.4f}, REF 2-GRAMS: {:.4f}, REF 3-GRAMS: {:.4f}".format(test_ref_ngrams[0], test_ref_ngrams[1], test_ref_ngrams[2]))
            f.write("GEN 1-GRAMS: {:.4f}, GEN 2-GRAMS: {:.4f}, GEN 3-GRAMS: {:.4f}".format(test_gen_ngrams[0], test_gen_ngrams[1], test_gen_ngrams[2]))
            f.write("BoW SIMs EXTREMA: {:.4f}, AVERAGE: {:.4f}, GREEDY: {:.4f}".format(test_bow_sims[0], test_bow_sims[1], test_bow_sims[2]))
            f.write("REF LEN: {:.4f}, GEN LEN: {:.4f}, SENT ACC: {:.4f}, SENT CORR: {:.4f}, REF IMP: {:.4f}, GEN IMP: {:.4f}".format(test_ref_lens, test_gen_lens, test_sentiment_accuracy, test_sentiment_correlation, test_ref_improvement, test_gen_improvement))
