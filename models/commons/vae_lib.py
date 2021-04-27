import os
import math
import time
import pprint
import random

from tqdm import tqdm
import dill as pickle
import numpy as np
from numpy import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

from utils import constant


def gumbel_softmax(logits, dim, tau=1.0):
    """
    Sample z ~ log p(z) + G(0, 1)
    """
    eps=1e-20
    noise = torch.rand(logits.size())
    noise = -torch.log(-torch.log(noise + eps) + eps) # gumble noise
    if constant.USE_CUDA:
        noise = noise.float().cuda()
    return F.softmax((logits + noise) / tau, dim=dim)

def reparameterization(mu, logvar, z_dim):
    """
    Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
    """
    eps = torch.randn(z_dim)
    eps = eps.cuda() if constant.USE_CUDA else eps
    return mu + torch.exp(logvar/2) * eps

def split_z(z, B, M, K):
    return z.view(B, M, K)

def merge_z(z, B, M, K):
    return z.view(B, M * K)

def cat_mi(p, q):
    pass
    
def cat_kl(logp, logq, dim=1):
    """
    \sum q * log(q/p)
    """
    if logq.dim() > 2:
        logq = logq.squeeze()

    q = torch.exp(logq)
    kl = torch.sum(q * (logq - logp), dim=dim)
    return torch.mean(kl)

def norm_kl(recog_mu, recog_logvar, prior_mu=None, prior_logvar=None):
    # find the KL divergence between two Gaussian distributions (defaults to standard normal for prior)
    if prior_mu is None:
        prior_mu = torch.zeros(1)
        prior_logvar = torch.ones(1)
    if constant.USE_CUDA:
        prior_mu = prior_mu.cuda()
        prior_logvar = prior_logvar.cuda()
    loss = 1.0 + (recog_logvar - prior_logvar)
    loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
    loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
    kl_loss = -0.5 * torch.mean(loss, dim=1)
    return torch.mean(kl_loss)
