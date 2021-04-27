import os
import math
import time
import pprint

from tqdm import tqdm
import dill as pickle
import numpy as np
from numpy import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils import constant, tile, text_input2bert_input, top_k_top_p_filtering
from models import RNNEncoder, RNNDecoder


class RLSeq(nn.Module):
    def __init__(self, encoder, decoder, vocab):
        super(RLSeq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_attn = True if constant.attn != "none" else False
        self.use_beam = constant.beam
        self.use_cuda = constant.USE_CUDA
        self.embeddings_cpu = constant.embeddings_cpu

        self.V = self.encoder.V
        self.H = self.encoder.H
        self.L = self.encoder.L
        self.D = self.encoder.D

        self.vocab = vocab

        if constant.bi == 'bi':
            self.reduce_state = nn.Linear(self.H * 2, self.H)

    def init_multitask(self):
        self.clf = nn.Linear(self.H, 1)

    # For Reward shaping 
    def init_aux_reward(self, reward_model):
        self.aux_reward = reward_model

    # For REINFORCE
    def init_baseline_reward(self):
        self.baseline_reward = nn.Linear(self.H, 1)

    # For REINFORCE
    def init_reward(self, reward_model):
        self.reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.reward = reward_model

    # For REINFORCE
    def init_user_model(self, user_model):
        self.user_model = user_model

    # For Curiosity Driven
    def init_intrinsic_curosity_module(self):
        # receive h_t and transform to \phi_t
        self.curiosity_encoder = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.SELU(),
            nn.Dropout(constant.dropout),
            nn.Linear(self.H, constant.CD),
            nn.SELU()
        )

        # receive h_t and h_{t+1} to predict a_t
        self.curiosity_inverse = nn.Linear(constant.CD * 2, self.V)

        # receive h_t and embedding(a_t) to predict h_{t+1}
        self.curiosity_forward = nn.Sequential(
            nn.Linear(constant.CD + self.D, self.H),
            nn.SELU(),
            nn.Dropout(constant.dropout),
            nn.Linear(self.H, constant.CD),
            nn.SELU(),
        )

    def encode(self, seqs, lens):
        src_h, src_h_0 = self.encoder(seqs, lens)

        # Init decoder hidden with encoder final hidden w/o attn
        dec_h_t = src_h_0.unsqueeze(0)
        
        if constant.bi == 'bi':
            src_h = self.reduce_state(src_h)
            dec_h_t = self.reduce_state(dec_h_t)

        return src_h, dec_h_t

    def decode(self, t, x_t, dec_h_t, src_h, targets, tau=1.0, sample=False, use_mle=False, min_dec_len=1):
        # y_t: B x V, dec_h_t: 1 x B x H
        y_t, dec_h_t = self.decoder(x_t, dec_h_t, src_h, self.use_attn)
        if use_mle:
            x_t = targets[:,t] # Next input is current target
        elif sample: # torch.multinomial sample vector B with input_dist y_t: B x V
            y_t[:, constant.unk_idx] = -float('Inf')
            # prevent empty string by setting min decoding length
            if t < min_dec_len: 
                y_t[:, constant.eou_idx] = -float('Inf')
            if constant.topk:
                filtered = top_k_top_p_filtering(y_t.data / tau, top_k=constant.topk_size)
                x_t = torch.multinomial(F.softmax(filtered, dim=1), 1).long().squeeze() # x_t: B
            else:
                x_t = torch.multinomial(F.softmax(y_t.data / tau, dim=1), 1).long().squeeze() # x_t: B
        else:
            y_t[:, constant.unk_idx] = -float('Inf')
            # prevent empty string by setting min decoding length
            if t < min_dec_len: 
                y_t[:, constant.eou_idx] = -float('Inf')
            if constant.topk:
                filtered = top_k_top_p_filtering(y_t.data / tau, top_k=constant.topk_size)
                x_t = torch.multinomial(F.softmax(filtered, dim=1), 1).long().squeeze() # x_t: B
            else:
                _, topi = y_t.data.topk(1)
                if self.use_cuda:
                  x_t = torch.cuda.LongTensor(topi.view(-1)) # Chosen word is next input
                else:
                  x_t = torch.LongTensor(topi.view(-1)) # Chosen word is next input
        # if self.use_cuda:
        #     x_t = x_t.cuda()

        return y_t, dec_h_t, x_t
    
    def forward(self, seqs, lens, targets, sentiments=None, test=False, use_mle=False):
        B, T = targets.shape

        src_h, dec_h_t = self.encode(seqs, lens)
        
        # clf
        if not test and constant.use_sentiment:
            clf_logits = self.clf(dec_h_t).squeeze()

        if dec_h_t.shape[0] == 1 and len(dec_h_t.shape) < 3:
            dec_h_t = dec_h_t.unsqueeze(0)
        x_t = torch.LongTensor([constant.sou_idx] * B) # Trigger Generation with SOS Token
        xs = torch.zeros(T, B).long()

        probs = torch.zeros(T, B, self.V)
        if not use_mle:
            # xs = torch.zeros(T, B).long() # decoded wrods
            rs = torch.zeros(T, B).float() # baseline rewards
            if constant.use_curiosity:
                # curiosity_forward_criterion = nn.MSELoss()
                curiosity_inverse_criterion = nn.CrossEntropyLoss(ignore_index=constant.eou_idx)
                curiosity_features = torch.zeros(T, B, constant.CD).float() 
                curiosity_forwards = torch.zeros(T, B, constant.CD).float() 
                curiosity_actions = torch.zeros(T, B, self.V).float()
                R_c = torch.zeros(T, B).float()
            step_mask = torch.ones(B).float()
            step_masks = []
            step_losses = []
        if self.use_cuda:
            x_t = x_t.cuda()
            xs = xs.cuda()
            probs = probs.cuda()
            if not use_mle:
                rs = rs.cuda()
                step_mask = step_mask.cuda()
                if constant.use_curiosity:
                    curiosity_features = curiosity_features.cuda()
                    curiosity_forwards = curiosity_forwards.cuda()
                    curiosity_actions = curiosity_actions.cuda()
                    R_c = R_c.cuda()

        dec_h_0, x_0 = dec_h_t, x_t

        prev_curiosity_feature = None
        # Run through decoder one time step at a time
        for t in range(T):
            y_t, dec_h_t, x_t = self.decode(t, x_t, dec_h_t, src_h, targets, tau=constant.tau, use_mle=False if test else use_mle, sample=False if test else (not use_mle))
            probs[t] = y_t
            xs[t] = x_t

            if not use_mle:
                # save the decoded sentence

                gold_probs = torch.gather(y_t, 1, x_t.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs)
                step_loss = step_loss * step_mask

                step_losses.append(step_loss)
                step_masks.append(step_mask)

                step_mask = torch.clamp(step_mask - (x_t == constant.eou_idx).float(), min=0.0)

                # calculate baseline rewards for each timestep (only update regression model - detach dec_h_t)
                if not test and constant.use_baseline:
                    rs[t] = self.baseline_reward(dec_h_t.squeeze().detach()).squeeze() * step_mask
                
                # curiosity features and forward model
                if not test and constant.use_curiosity:
                    curiosity_feature = self.curiosity_encoder(dec_h_t.squeeze().detach()) * step_mask.unsqueeze(1)
                    curiosity_features[t] = curiosity_feature
                    curiosity_forward = self.curiosity_forward(torch.cat([curiosity_feature, self.decoder.embedding(xs[t]).detach()], dim=-1)) * step_mask.unsqueeze(1)
                    curiosity_forwards[t] = curiosity_forward
                    # curiosity inverse model <- from a_1 to a_t-1
                    if t > 0:
                        curiosity_actions[t-1] = self.curiosity_inverse(torch.cat((prev_curiosity_feature, curiosity_feature), dim=-1)) * step_mask.unsqueeze(1)
                        R_c[t-1] = 0.5 * torch.pow((curiosity_feature - curiosity_forward).norm(p=2, dim=-1), 2) * step_mask
                        prev_curiosity_feature = curiosity_feature
                    else:
                        prev_curiosity_feature = curiosity_feature

        

        # curiosity reward is MSE loss of || \phi_t || given h_{t-1} <- from a_1 to a_t-1
        if not use_mle and not test and constant.use_curiosity:
            # R_c = curiosity_forward_criterion(R_c, curiosity_forwards[:, 1:]) # don't predict first state
            # L_i = curiosity_inverse_criterion(curiosity_actions.transpose(0, 1).contiguous().view(B*(T-1), -1), xs.transpose(0, 1).contiguous()[:, :-1].contiguous().view(B*(T-1)) ) # don't predict last action
            # curiosity_actions[T-1] = self.curiosity_inverse(torch.cat((curiosity_features[-1], torch.zeros(B, 128)), dim=-1)) * step_mask.unsqueeze(1)
            last_action = torch.zeros(B, self.V).float().cuda() if constant.USE_CUDA else torch.zeros(B, self.V).float()
            # eou_idx = torch.LongTensor([constant.eou_idx] * B)
            # if constant.USE_CUDA:
            #     last_action = last_action.cuda()
                # eou_idx = eou_idx.cuda()
            # last_action.scatter_(1, eou_idx.unsqueeze(1), float('Inf'))
            # last_action *= step_mask.unsqueeze(1)
            curiosity_actions[T-1] = last_action
            L_i = curiosity_inverse_criterion(curiosity_actions.transpose(0, 1).contiguous().view(B*T, -1), xs.transpose(0, 1).contiguous().view(B*T) ) # don't predict last action

        # generate from the decoded sentence
        xs = xs.cpu().data.numpy().T # B, T
        # iter(callable, sentinel) break loop when sentinel is hit
        sentences = [" ".join([self.vocab.index2word[x_t] for x_t in iter(lambda x=iter(gens): next(x), constant.eou_idx)]) for gens in xs] 

        if use_mle:
            return probs, sentences
        else:
            if constant.use_bert:
                if constant.use_user:
                    contexts = [" ".join([self.vocab.index2word[x_t] for x_t in iter(lambda x=iter(seq): next(x), constant.pad_idx)]) for seq in seqs.cpu().data.numpy()] 
                    sents = [context + ' ' + sent for context, sent in zip(contexts, sentences)]

                    sents = [self.vocab.transform_one(sent) for sent in sents]
                    lens = [len(sentence) for sentence in sents]
                    sort = np.argsort(lens)[::-1].tolist()
                    unsort = np.argsort(sort).tolist()
                    sents = np.array(sents, dtype='object')[sort].tolist()
                    lens = np.array(lens)[sort]#.tolist()
                    # Pad dialogs and targets to their respective max batch lens
                    B = len(sentences)
                    L = lens[0]
                    padded_sentences = torch.ones((B, L)) * constant.pad_idx
                    for b in range(B):
                        padded_sentences[b, :lens[b]] = torch.from_numpy(np.array(sents[b]))
                
                    padded_sentences = padded_sentences.long()
                    if constant.USE_CUDA:
                        padded_sentences = padded_sentences.cuda()

                    user_sents = np.array(self.user_model.predict_batch(padded_sentences, lens, np.zeros((B, L))))[unsort].tolist()
                    R = self.get_reward(user_sents)
                else:
                    R = self.get_reward(sentences)
            else:
                sents = sentences
                if constant.use_context:
                    contexts = [" ".join([self.vocab.index2word[x_t] for x_t in iter(lambda x=iter(seq): next(x), constant.pad_idx)]) for seq in seqs.cpu().data.numpy()] 
                    sents = [context + ' ' + sent for context, sent in zip(contexts, sents)]

                sents = [self.vocab.transform_one(sent) for sent in sents]
                lens = [len(sentence) for sentence in sents]
                sort = np.argsort(lens)[::-1].tolist()
                unsort = np.argsort(sort).tolist()
                sents = np.array(sents, dtype='object')[sort].tolist()
                lens = np.array(lens)[sort]#.tolist()
                # Pad dialogs and targets to their respective max batch lens
                B = len(sentences)
                L = lens[0]
                padded_sentences = torch.ones((B, L)) * constant.pad_idx
                for b in range(B):
                    padded_sentences[b, :lens[b]] = torch.from_numpy(np.array(sents[b]))
            
                padded_sentences = padded_sentences.long()
                if constant.USE_CUDA:
                    padded_sentences = padded_sentences.cuda()

                # get reward with generated sentence
                with torch.no_grad():
                    R = self.reward.predict_prob(padded_sentences, lens)[unsort]

            step_masks = torch.stack(step_masks, dim=1).float()
            dec_lens_var = torch.sum(step_masks, dim=1)

            if not test and constant.use_self_critical:
                # decode greedily without teacher forcing or sampling
                dec_h_t, x_t, src_h = dec_h_0.detach(), x_0.detach(), src_h.detach()
                with torch.no_grad():
                    for t in range(T):
                        y_t, dec_h_t, x_t = self.decode(t, x_t, dec_h_t, src_h, targets, use_mle=False, sample=False)
                        probs[t] = y_t
                greedy_sents = self.greedy_search(probs, self.vocab)
                input_ids, input_masks, segment_ids = zip(*[text_input2bert_input(sentence, self.reward_tokenizer, seq_length=128) for sentence in greedy_sents])
                input_ids = torch.stack(input_ids)
                input_masks = torch.stack(input_masks)
                segment_ids = torch.stack(segment_ids)

                if constant.USE_CUDA:
                    input_ids = input_ids.cuda()
                    input_masks = input_masks.cuda()
                    segment_ids = segment_ids.cuda()

                # get reward with generated sentence
                with torch.no_grad():
                    R_g = self.reward.predict_prob((input_ids, segment_ids, input_masks))

                if not test and constant.use_emotion:
                    return torch.stack(step_losses, 1), dec_lens_var, R_g, R, greedy_sents, sentences, clf_logits
                return torch.stack(step_losses, 1), dec_lens_var, R_g, R, greedy_sents, sentences
            
            elif not test and constant.use_sentiment and constant.use_sentiment_agreement:
                # R = R - sentiments.unsqueeze(1)
                contexts = [" ".join([self.vocab.index2word[x_t] for x_t in iter(lambda x=iter(seq): next(x), constant.pad_idx)]) for seq in seqs.cpu().data.numpy()] 
                R = R - self.get_reward(contexts)
                return torch.stack(step_losses, 1), dec_lens_var, rs, R, sentences, clf_logits

            elif not test and constant.use_sentiment_agreement:
                # R = R - sentiments.unsqueeze(1)
                contexts = [" ".join([self.vocab.index2word[x_t] for x_t in iter(lambda x=iter(seq): next(x), constant.pad_idx)]) for seq in seqs.cpu().data.numpy()] 
                R = R - self.get_reward(contexts)
                return torch.stack(step_losses, 1), dec_lens_var, rs, R, sentences

            elif not test and constant.use_sentiment and constant.aux_reward_model != '':
                # get sentiment for current context generation => sentiments
                # get current generation => already given
                # get reward with generated sentence
                with torch.no_grad():
                    gen_sentiments = torch.sigmoid(self.aux_reward.predict_prob((input_ids, segment_ids, input_masks)).squeeze())

                ctx_sentiments = torch.sigmoid(clf_logits.squeeze()).detach()
                R_s = ctx_sentiments - gen_sentiments
                # R_s = sentiments - gen_sentiments
                # R_s = -torch.abs(R_s)
                R_s = -torch.pow(R_s, 2)# + 0.5
                # R_s *= 2

                # R_s = torch.ones(B) * -1
                # if constant.USE_CUDA:
                #     R_s = R_s.cuda()
                # R_s[sentiments == gen_sentiments.float()] = 1.0
                return torch.stack(step_losses, 1), dec_lens_var, rs, R, R_s.unsqueeze(1), sentences, clf_logits

            elif not test and constant.use_arl:
                targets = [" ".join([self.vocab.index2word[b] 
                                for b in a
                                if b != constant.eou_idx])
                            for a in targets.cpu().data.numpy()]
                input_ids, input_masks, segment_ids = zip(*[text_input2bert_input(sentence, self.reward_tokenizer, seq_length=128) for sentence in targets])
                input_ids = torch.stack(input_ids)
                input_masks = torch.stack(input_masks)
                segment_ids = torch.stack(segment_ids)

                if constant.USE_CUDA:
                    input_ids = input_ids.cuda()
                    input_masks = input_masks.cuda()
                    segment_ids = segment_ids.cuda()

                # get reward with generated sentence
                with torch.no_grad():
                    arl = self.reward.predict_prob((input_ids, segment_ids, input_masks))
                    arl = torch.clamp(arl, min=0.5)
                    # arl = torch.abs((torch.ones(arl.size()) * 0.5).to(arl.device) - arl) * 2

                return torch.stack(step_losses, 1), dec_lens_var, rs, R, arl, sentences

            elif not test and constant.use_sentiment:
                return torch.stack(step_losses, 1), dec_lens_var, rs, R, sentences, clf_logits

            elif not test and constant.use_curiosity:
                return torch.stack(step_losses, 1), dec_lens_var, rs, R, R_c, L_i, sentences
            
            else:
                return torch.stack(step_losses, 1), dec_lens_var, rs, R, sentences

    def get_reward(self, sentences):
        input_ids, input_masks, segment_ids = zip(*[text_input2bert_input(sentence, self.reward_tokenizer, seq_length=128) for sentence in sentences])
        input_ids = torch.stack(input_ids)
        input_masks = torch.stack(input_masks)
        segment_ids = torch.stack(segment_ids)

        if constant.USE_CUDA:
            input_ids = input_ids.cuda()
            input_masks = input_masks.cuda()
            segment_ids = segment_ids.cuda()

        # get reward with generated sentence
        with torch.no_grad():
            R = self.reward.predict_prob((input_ids, segment_ids, input_masks))

        return R

    def greedy_search(self, probs, vocab):
        hyp = []
        # words: T x B x V => T x B => B x T
        words = np.array([prob.cpu().data.topk(1)[1].squeeze().numpy() for prob in probs]).T
        words = [[vocab.index2word[b] for b in a] for a in words]
        for _, row in enumerate(words): # [B, T]
            st = ''
            for word in row:
                if word == '__eou__' or word == '__pad__':
                    break
                else:
                    st += word + ' '
            hyp.append(str(st.lstrip().rstrip()))
        
        return hyp

    def decode_one(self, t, x_t, dec_h_t, src_h, tau=1.0, min_dec_len=1):
        # y_t: B x V, dec_h_t: 1 x B x H
        y_t, dec_h_t = self.decoder.predict_one(x_t, dec_h_t, src_h, self.use_attn)
        if constant.topk: # torch.multinomial sample vector B with input_dist y_t: B x V
            y_t[constant.unk_idx] = -float('Inf')
            # prevent empty string by setting min decoding length
            if t < min_dec_len: 
                y_t[constant.eou_idx] = -float('Inf')
            filtered = top_k_top_p_filtering(y_t.data / tau, top_k=constant.topk_size)
            x_t = torch.multinomial(F.softmax(filtered, dim=0), 1).long() # x_t: B
        else:
            y_t[constant.unk_idx] = -float('Inf')
            # prevent empty string by setting min decoding length
            if t < min_dec_len:
                y_t[constant.eou_idx] = -float('Inf')
            _, topi = y_t.data.topk(1)
            if self.use_cuda:
                x_t = torch.cuda.LongTensor(topi.view(-1)) # Chosen word is next input
            else:
                x_t = torch.LongTensor(topi.view(-1)) # Chosen word is next input

        return y_t, dec_h_t, x_t

    def predict_one(self, seq, max_dec_len=30):
        with torch.no_grad():
            src_h, dec_h_t = self.encoder.predict_one(seq)
            dec_h_t = dec_h_t.unsqueeze(0)
            if constant.bi == 'bi':
                src_h = self.reduce_state(src_h)
                dec_h_t = self.reduce_state(dec_h_t)
            x_t = torch.LongTensor([constant.sou_idx]) # Trigger Generation with SOS Token
            xs = torch.zeros(max_dec_len).long()

            # Run through decoder one time step at a time
            for t in range(max_dec_len):
                _, dec_h_t, x_t = self.decode_one(t, x_t, dec_h_t, src_h, tau=constant.tau)
                xs[t] = x_t

            return xs

    def predict_batch(self, seqs, lens, targets):
        with torch.no_grad():
            _, sentences = self.forward(seqs, lens, np.zeros((len(seqs), lens[0])), test=True)
            return sentences
