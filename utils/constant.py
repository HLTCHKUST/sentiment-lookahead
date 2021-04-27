import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="RNN") # RNN, LVED

# Hyperparams
parser.add_argument("--C", type=int, default=1) # number of classes
parser.add_argument("--H", type=int, default=300)
parser.add_argument("--D", type=int, default=300)
parser.add_argument("--B", type=int, default=32)
parser.add_argument("--L", type=int, default=1)
parser.add_argument("--M", type=int, default=3) # number of latent variables
parser.add_argument("--K", type=int, default=5) # dimension of latent variable
parser.add_argument("--CD", type=int, default=256) # dimension of ICM
parser.add_argument("--beta", type=float, default=0.5) # aux reward lambda
parser.add_argument("--lambda_aux", type=float, default=0.5) # aux reward lambda
parser.add_argument("--lambda_emo", type=float, default=0.5) # emo loss lambda
parser.add_argument("--lambda_gen", type=float, default=0.5) # gen loss lambda
parser.add_argument("--lambda_mle", type=float, default=0.5) # mle loss lambda
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--tau", type=float, default=1.0) # softmax temperature
parser.add_argument("--bi", type=str, default="none") # none, bi
parser.add_argument("--mlp", action="store_true")
parser.add_argument("--lstm", action="store_true")
parser.add_argument("--dropout", type=float, default=0.5)

# Train Settings
parser.add_argument("--attn", type=str, default="none") # none, dot, concat (luong), general
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--optim", type=str, default="Adam") # Adam, SGD
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--round_robin", action="store_true")
parser.add_argument("--parse", type=str, default="none") # none, user, system
parser.add_argument("--eval_parse", action="store_true") # eval as parse or not
parser.add_argument("--embedding", type=str, default="random") # random, fasttext
parser.add_argument("--share_rnn", action="store_true")
parser.add_argument("--weight_tie", action="store_true")
parser.add_argument("--embeddings_cpu", action="store_true")
parser.add_argument("--share_embeddings", action="store_true")
parser.add_argument("--update_embeddings", action="store_true")

# Beam Search
parser.add_argument("--beam", action="store_true")
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--topk", action="store_true")
parser.add_argument("--topk_size", type=int, default=40)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--max_enc_steps", type=int, default=400)
parser.add_argument("--max_dec_steps", type=int, default=20)
parser.add_argument("--min_dec_steps", type=int, default=5)

## Data & Task: Single vs Mutli
parser.add_argument("--data", type=str, default="dailydialog") # "dailydialog", "empathetic-dialogue", "personachat", "ed+dd", "all", "sst"
parser.add_argument("--eval_data", type=str, default="empathetic-dialogue") # "dailydialog", "empathetic-dialogue", "personachat", "ed+dd", "all", "sst"
parser.add_argument("--task", type=str, default="emotion") # "emotion", "sentiment", "seq2seq", "multiseq", "rlseq", "lved"
parser.add_argument("--split", type=str, default="dev") # train, dev, test 
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--discrete", action="store_true") # use emotion_t. Otherwise, emtion_(t+1)
parser.add_argument("--use_arl", action="store_true") # Auto-tune RL
parser.add_argument("--use_baseline", action="store_true") # baseline reward
parser.add_argument("--use_binary", action="store_true") # use binary traces
parser.add_argument("--use_bpr", action="store_true") # batch prior regularization
parser.add_argument("--use_bow", action="store_true") # BoW loss
parser.add_argument("--use_bert", action="store_true") # Use pre-trained BERT for sentiment
parser.add_argument("--use_context", action="store_true")
parser.add_argument("--use_current", action="store_true")
parser.add_argument("--use_curiosity", action="store_true") # curiosity reward
parser.add_argument("--use_cycle", action="store_true") # cycle consistency
parser.add_argument("--use_emotion", action="store_true")
parser.add_argument("--use_hybrid", action="store_true") # use hybrid loss
parser.add_argument("--use_lang", action="store_true")
parser.add_argument("--use_kl_anneal", action="store_true")
parser.add_argument("--use_sentiment", action="store_true")
parser.add_argument("--use_sentiment_agreement", action="store_true")
parser.add_argument("--use_self_critical", action="store_true") # use self critical baseline
parser.add_argument("--use_topic", action="store_true") # use topic info for LVED
parser.add_argument("--use_tau_anneal", action="store_true")
parser.add_argument("--use_user", action="store_true") # use user simulation
parser.add_argument("--pretrain_curiosity", action="store_true")
parser.add_argument("--reset_linear", action="store_true")
parser.add_argument("--conditional_vae", action="store_true")
parser.add_argument("--grid_search", action="store_true")

# Save/Load
parser.add_argument("--restore", action="store_true")
parser.add_argument("--restore_path", type=str, default="") 
parser.add_argument("--test", action="store_true")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--lang_path", type=str, default="") # '_shared' vs ''
parser.add_argument("--policy_model", type=str, default="")
parser.add_argument("--reward_model", type=str, default="")
parser.add_argument("--user_model", type=str, default="")
parser.add_argument("--aux_reward_model", type=str, default="")
parser.add_argument("--sentiment_clf", type=str, default="")


arg = parser.parse_args()
print(arg)
model = arg.model

# Hyperparameters
C = arg.C
H = arg.H
D = arg.D
B = arg.B
L = arg.L
M = arg.M
K = arg.K
CD = arg.CD
beta = arg.beta
lambda_aux = arg.lambda_aux
lambda_emo = arg.lambda_emo
lambda_gen = arg.lambda_gen
lambda_mle = arg.lambda_mle
bi=arg.bi
lr=arg.lr
tau = arg.tau
mlp = arg.mlp
lstm = arg.lstm
beam_size = arg.beam_size
topk = arg.topk
topk_size = arg.topk_size

attn = arg.attn
beam = arg.beam
optim = arg.optim
parse = arg.parse
eval_parse = arg.eval_parse
dropout = arg.dropout
embedding = arg.embedding
round_robin = arg.round_robin
epochs = arg.epochs
share_rnn = arg.share_rnn
weight_tie = arg.weight_tie
embeddings_cpu = arg.embeddings_cpu
share_embeddings = arg.share_embeddings
update_embeddings = arg.update_embeddings

rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

max_grad_norm = arg.max_grad_norm
max_enc_steps = arg.max_enc_steps
max_dec_steps = arg.max_dec_steps
min_dec_steps = arg.min_dec_steps

USE_CUDA = arg.cuda

unk_idx = 0
pad_idx = 1
sou_idx = 2
eou_idx = 3

data = arg.data
eval_data = arg.eval_data
task = arg.task
split = arg.split
shuffle = arg.shuffle
discrete = arg.discrete
use_arl = arg.use_arl
use_baseline = arg.use_baseline
use_bpr = arg.use_bpr
use_bow = arg.use_bow
use_bert = arg.use_bert
use_cycle = arg.use_cycle
use_curiosity = arg.use_curiosity
use_lang = arg.use_lang
use_topic = arg.use_topic
use_binary = arg.use_binary
use_current = arg.use_current
use_context = arg.use_context
use_hybrid = arg.use_hybrid
use_emotion = arg.use_emotion
use_sentiment = arg.use_sentiment
use_sentiment_agreement = arg.use_sentiment_agreement
use_self_critical = arg.use_self_critical
use_user = arg.use_user
reset_linear = arg.reset_linear
use_kl_anneal = arg.use_kl_anneal
use_tau_anneal = arg.use_tau_anneal
pretrain_curiosity = arg.pretrain_curiosity
conditional_vae = arg.conditional_vae
grid_search = arg.grid_search

restore = arg.restore
restore_path = arg.restore_path

test = arg.test
test_path = arg.test_path
lang_path = arg.lang_path
policy_model = arg.policy_model
reward_model = arg.reward_model
user_model = arg.user_model
aux_reward_model = arg.aux_reward_model
sentiment_clf = arg.sentiment_clf