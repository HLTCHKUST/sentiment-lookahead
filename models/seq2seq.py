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

from utils import Beam, GNMTGlobalScorer, constant, tile, top_k_top_p_filtering
from models import RNNEncoder, RNNDecoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_attn = True if constant.attn != "none" else False
        self.use_beam = constant.beam
        self.use_cuda = constant.USE_CUDA
        self.embeddings_cpu = constant.embeddings_cpu
        self.teacher_forcing_ratio = 0.5

        self.vocab = vocab

        self.V = self.encoder.V
        self.H = self.encoder.H
        self.L = self.encoder.L
        self.D = self.encoder.D

        if constant.bi == 'bi':
            self.reduce_state = nn.Linear(self.H * 2, self.H)

    def encode(self, seqs, lens):
        src_h, src_h_0 = self.encoder(seqs, lens)

        # Init decoder hidden with encoder final hidden w/o attn
        dec_h_t = src_h_0.unsqueeze(0)
        
        if constant.bi == 'bi':
            src_h = self.reduce_state(src_h)
            dec_h_t = self.reduce_state(dec_h_t)

        return src_h, dec_h_t

    def decode(self, t, x_t, dec_h_t, src_h, targets, use_teacher_forcing):
        # y_t: B x V, dec_h_t: 1 x B x H
        y_t, dec_h_t = self.decoder(x_t, dec_h_t, src_h, self.use_attn)
        if use_teacher_forcing:
            x_t = targets[:,t] # Next input is current target
        else:
            if constant.topk:
                filtered = top_k_top_p_filtering(y_t.data / constant.tau, top_k=constant.topk_size)
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
    
    def forward(self, seqs, lens, targets, test=False):
        batch_size, max_target_len = targets.shape

        src_h, dec_h_t = self.encode(seqs, lens)
        if dec_h_t.shape[0] == 1 and len(dec_h_t.shape) < 3:
            dec_h_t = dec_h_t.unsqueeze(0)

        x_t = torch.LongTensor([constant.sou_idx] * batch_size) # Trigger Generation with SOS Token
        xs = torch.zeros(max_target_len, batch_size)

        probs = torch.zeros(max_target_len, batch_size, self.V)
        if self.use_cuda:
            probs = probs.cuda()
            x_t = x_t.cuda()
            xs = xs.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = True #random.random() < self.teacher_forcing_ratio
        if test: use_teacher_forcing = False

        # Run through decoder one time step at a time
        for t in range(max_target_len):
            y_t, dec_h_t, x_t = self.decode(t, x_t, dec_h_t, src_h, targets, use_teacher_forcing)
            probs[t] = y_t
            xs[t] = x_t

        if test:
            xs = xs.cpu().data.numpy().T # B, T
            sentences = [" ".join([self.vocab.index2word[x_t] for x_t in iter(lambda x=iter(gens): next(x), constant.eou_idx)]) for gens in xs] 
            return probs, sentences
        
        return probs

    def predict_one(self):
        pass
    
    def predict_batch(self, seqs, lens, targets):
        with torch.no_grad():
            _, sentences = self.forward(seqs, lens, targets, test=True)
            return sentences

    def greedy_search(self, probs, vocab):
        hyp = []
        # words: T x B x V => T x B => B x T
        words = np.array([prob.cpu().data.topk(1)[1].squeeze().numpy() for prob in probs]).T
        words = [[vocab.index2word[b] for b in a] for a in words]
        for i, row in enumerate(words): # [B, T]
            st = ''
            for word in row:
                if word == '__eou__' or word == '__pad__':
                    break
                else:
                    st += word + ' '
            hyp.append(str(st.lstrip().rstrip()))
        
        return hyp


    def topk_search(self, probs, vocab):
        hyp = []
        # words: T x B x V => T x B => B x T
        # words = np.array([prob.cpu().data.topk(1)[1].squeeze().numpy() for prob in probs]).T
        
        words = np.array(
            [torch.multinomial(
                F.softmax(
                    top_k_top_p_filtering(prob / constant.tau, top_k=constant.topk_size)
                    , dim=1)
                , 1).long().squeeze().cpu().data.numpy()
                for prob in probs]).T
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


    def beam_search(self, seqs, lens, B, L, vocab):
        n_best = 1
        K = constant.beam_size
        beam = [Beam(K, 
                     n_best=n_best,
                     global_scorer=GNMTGlobalScorer(),)
                     # min_length=self.min_length,                                               
                     # stepwise_penalty=self.stepwise_penalty,                                                       
                     # block_ngram_repeat=self.block_ngram_repeat,                                                            
                     # exclusion_tokens=exclusion_tokens)                                         
                for _ in range(B)]

        # (1) Run the encoder on the src.
        src_h, dec_h_t = self.encode(seqs, lens)
        # print(src_h.shape) # B, L, H
        # print(dec_h_t.shape) # 1, B, H

        # (2) Repeat src objects `beam_size` times.
        # Tile states and inputs K times
        dec_h_t = tile(dec_h_t, K, dim=1)
        # print(dec_h_t.shape) # 1, B*K, H
        
        if self.use_attn:
            # print(src_h.shape) # [B, L, H]
            src_h = tile(src_h.contiguous(), K, dim=0)
            # print(src_h.shape) # [B*K, L, H]
        
        # We use now  batch_size x beam_size (same as fast mode)

        # (3) run the decoder to generate sentences, using beam search.
        for t in range(L):
            if all((b.done() for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            x_t = torch.stack([b.get_current_state() for b in beam])#.t().contiguous()
            x_t = x_t.view(-1)

            # (b) Decode and forward
            # y_t: B*K x V, dec_h_t: 1 x B*K x H
            y_t, dec_h_t = self.decoder(x_t, dec_h_t, src_h)
            y_t = y_t.view(B, K, -1) # B, K, V
            y_t = F.log_softmax(y_t, dim=2)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beams (beam search per sequence)
            for j, b in enumerate(beam):
                b.advance(y_t[j], None)
                select_indices_array.append(
                    b.get_current_origin() + j * K)
            
            select_indices = torch.cat(select_indices_array)
            dec_h_t = dec_h_t.index_select(1, select_indices) # select correct nodes

        # (4) Extract sentences from beam.
        preds = []
        for b in beam:
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for times, k in ks[:n_best]:
                hyp, _ = b.get_hyp(times, k)
                hyps.append(" ".join([vocab.index2word[word.item()] for word in hyp if word.item() not in [constant.eou_idx, constant.pad_idx]]))
            preds.append(hyps[0])
        return preds
