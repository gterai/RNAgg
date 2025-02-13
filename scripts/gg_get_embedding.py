import sys
import os
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
#sys.path.append("Users/terai/RNAVAE")
#import basic
import pickle
#import copy
import numpy as np
#import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import RNAgg_VAE
import SS2shape3
import Binary_matrix
import utils_gg as utils

#import graphviz
from torch.utils.data import DataLoader
#from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from joblib import Parallel, delayed
import random
#from libpysal.weights import KNN

NUC_LETTERS = list('ACGU-x')
#SS_LETTERS = list('.()')
G_DIM=11

def main(args: dict):
    
    # 準備
    token2idx = utils.get_token2idx(NUC_LETTERS)
    idx2token = dict([y,x] for x,y in token2idx.items())
    word_size = len(NUC_LETTERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, file=sys.stderr)
    
    # モデルの初期化とロード
    checkpoint = torch.load(args.model)
    #d_rep, max_len = checkpoint['d_rep'], checkpoint['max_len']
    d_rep, max_len, model_type, nuc_yes_no = checkpoint['d_rep'], checkpoint['max_len'], checkpoint['type'], checkpoint['nuc_only']
    if model_type == 'act':
        if nuc_yes_no == 'yes':
            model = RNAgg_VAE.MLP_VAE_REGRE(max_len*(word_size), max_len*(word_size), d_rep, device=device).to(device)
        else:
            model = RNAgg_VAE.MLP_VAE_REGRE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), d_rep, device=device).to(device)
    else:
        if nuc_yes_no == 'yes':
            model = RNAgg_VAE.MLP_VAE(max_len*(word_size), max_len*(word_size), d_rep, device=device).to(device)
        else:
            model = RNAgg_VAE.MLP_VAE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), d_rep, device=device).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 学習データの分散表現を取得する
    sid2seq, sid2ss  = utils.readInput(args.input)
    sid_list = list(sid2seq.keys())
    sid2act = dict([(sid, np.nan) for sid in sid_list]) #utils.Datasetを使うためのダミーデータ
    act_list = [sid2act[sid] for sid in sid_list]
    B_mat = Binary_matrix.makeMatrix(sid2seq, sid2ss, sid_list, word_size, G_DIM, token2idx, nuc_yes_no)

    d = utils.Dataset(B_mat, sid_list, act_list)
    train_dataloader = DataLoader(d, batch_size=args.s_bat, shuffle=False)
    sid_list_exec = []
    for x, t, v in train_dataloader:
        x = x.to(device)
        mean, var = model.encoder(x)
        #var *= 0
        #z_bat = model.reparameterize(mean, var)
        z_bat = mean
        if 'z' in locals():
            z = torch.cat((z, z_bat), dim=0)
        else:
            z = z_bat
        print(z.shape, file=sys.stderr)
        sid_list_exec += t

    #random.shuffle(sid_list_exec)
    z_np = z.to('cpu').detach().numpy().copy()
    
    emb_inf = {"sid_list":sid_list_exec, "emb":z_np}
    
    f = open(args.out_pkl, "wb")
    pickle.dump(emb_inf, f)
    f.close()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file name')
    parser.add_argument('model', help='trained VAE model')
    parser.add_argument('out_pkl', help='output pkl file')
    #parser.add_argument('act', help='activity table')
    #parser.add_argument('--regre', action='store_true', help='use regression combined model')
    #parser.add_argument('--k', type=int, default=1, help='number of nearest neibours')
    #parser.add_argument('--png', default="latent_act.png", help='output png file name')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')    
    #parser.add_argument('--nuc_only', action='store_true', help='nucleotide only model')
    #parser.add_argument('--plot_size', type=int, default=5, help='size of point')
    
    args = parser.parse_args()

    main(args)

