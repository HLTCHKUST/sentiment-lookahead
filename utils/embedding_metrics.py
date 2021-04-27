import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as cosine
from collections import Counter

class EmbeddingSim:
    """
    """
    def __init__(self, word2vec):
        """
        :param word2vec - a numpy array of word2vec with shape [vocab_size x emb_size]
        """
        super(EmbeddingSim, self).__init__()
        self.word2vec = word2vec
        
    def embedding(self, seqs): 
        """
        A numpy version of embedding
        :param seqs - ndarray [batch_sz x seqlen]
        """
        batch_size, seqlen = seqs.shape
        seqs = np.reshape(seqs, (-1)) # convert to 1-d indexes [(batch_sz*seqlen)]
        embs = self.word2vec[seqs] # lookup [(batch_sz*seqlen) x emb_sz]
        embs = np.reshape(embs, (batch_size, seqlen, -1)) # recover the shape [batch_sz x seqlen x emb_sz]
        return embs
    
    def extrema(self, embs, lens): # embs: [batch_size x seq_len x emb_size]  lens: [batch_size]
        """
        computes the value of every single dimension in the word vectors which has the greatest
        difference from zero.
        :param seq: sequence
        :param seqlen: length of sequence
        """
        # Find minimum and maximum value for every dimension in predictions
        batch_size, seq_len, emb_size = embs.shape
        max_mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
        for i,length in enumerate(lens):
            max_mask[i,:length,:]=1
        min_mask = 1-max_mask
        seq_max = (embs*max_mask).max(1) # [batch_sz x emb_sz]
        seq_min = (embs+min_mask).min(1)
        # Find the maximum absolute value in min and max data
        comp_mask = seq_max >= np.abs(seq_min)# [batch_sz x emb_sz]
        # Add vectors for finding final sequence representation for predictions
        extrema_emb = seq_max* comp_mask + seq_min* np.logical_not(comp_mask)
        return extrema_emb
    
    def mean(self, embs, lens):
        batch_size, seq_len, emb_size=embs.shape
        mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
        for i,length in enumerate(lens):
            mask[i,:length,:]=1
        return (embs*mask).sum(1)/(mask.sum(1)+1e-8)

    def sim_bow(self, pred, pred_lens, ref, ref_lens):
        """
        :param pred - ndarray [batch_size x seqlen]
        :param pred_lens - list of integers
        :param ref - ndarray [batch_size x seqlen]
        """
        # look up word embeddings for prediction and reference
        emb_pred = self.embedding(pred) # [batch_sz x seqlen1 x emb_sz]
        emb_ref = self.embedding(ref) # [batch_sz x seqlen2 x emb_sz]
        
        ext_emb_pred=self.extrema(emb_pred, pred_lens)
        ext_emb_ref=self.extrema(emb_ref, ref_lens)
        bow_extrema=cosine(ext_emb_pred, ext_emb_ref) # [batch_sz_pred x batch_sz_ref]
        
        avg_emb_pred = self.mean(emb_pred, pred_lens) # Calculate mean over seq
        avg_emb_ref = self.mean(emb_ref, ref_lens) 
        bow_avg = cosine(avg_emb_pred, avg_emb_ref) # [batch_sz_pred x batch_sz_ref]
        
        batch_pred, seqlen_pred, emb_size=emb_pred.shape
        batch_ref, seqlen_ref, emb_size=emb_ref.shape
        cos_sim = cosine(emb_pred.reshape((-1, emb_size)), emb_ref.reshape((-1, emb_size))) # [(batch_sz*seqlen1)x(batch_sz*seqlen2)]
        cos_sim = cos_sim.reshape((batch_pred, seqlen_pred, batch_ref, seqlen_ref))
        # Find words with max cosine similarity
        max12 = cos_sim.max(1).mean(2) # max over seqlen_pred
        max21 = cos_sim.max(3).mean(1) # max over seqlen_ref
        bow_greedy=(max12+max21)/2 # [batch_pred x batch_ref(1)]
        return np.max(bow_extrema), np.max(bow_avg), np.max(bow_greedy)
    