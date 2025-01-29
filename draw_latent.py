import sys
import os
import argparse
import copy
import numpy as np
import RNAgg_VAE
import SS2shape3
import matplotlib.pyplot as plt
import utils
import Binary_matrix
import pickle

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


NUC_LETTERS = list('ACGU-x')
G_DIM = 11 # 文法を格納する部分の次元

def main(args: dict):
    
    # オプションのチェック
    checkArgs(args)
    token2idx = utils.get_token2idx(NUC_LETTERS)
    #idx2token = dict([y,x] for x,y in token2idx.items())
    #print(token2idx) # vocablaryのチェック
    #print(idx2token) # vocablaryのチェック
    
    sid2seq, sid2ss  = utils.readInput(args.input)
    #max_len = max([len(x) for x in sid2seq.values()])
    #print(f"Maximul_length={max_len}")
    
    sid_list = list(sid2seq.keys())
    word_size = len(NUC_LETTERS)  # 塩基の種類
    VDIM = word_size + G_DIM # バイナリベクトルの次元


    if args.act_fname != None: # activity dataが指定されていれば得る
        sid2act = utils.readAct(args.act_fname)
    else: # 指定されていない場合には、ダミー値を入れる
        sid2act = dict([(sid, np.nan) for sid in sid_list])
        
        
    checkpoint = torch.load(args.model)
    d_rep, max_len, model_type, nuc_yes_no = checkpoint['d_rep'], checkpoint['max_len'], checkpoint['type'], checkpoint['nuc_only']
    print(f"model type:", model_type, file=sys.stderr)

    
    B_mat = Binary_matrix.makeMatrix(sid2seq, sid2ss, sid_list, word_size, G_DIM, token2idx, nuc_yes_no)
    
    
    act_list = [sid2act[sid] for sid in sid_list]
    d = utils.Dataset(B_mat, sid_list, act_list)
    train_dataloader = DataLoader(d, batch_size=args.s_bat, shuffle=False) # 同じUMAPが再現されるようにFalseにしてある。
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, file=sys.stderr)




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

    # モデルのロード
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    umap_model = draw_latent(args.image, model, train_dataloader, sid2seq, sid2act, device, args)

    # umapモデルの保存
    if args.save_umap:
        with open(args.save_umap, "wb") as f:
            pickle.dump(umap_model, f)
            
def draw_latent(png_fname, model, train_dataloader, sid2seq, sid2act, device, args):

    c = []
    for (x,t,v) in train_dataloader:
        x = x.to(device)
        s = x.shape
        
        mean, var = model.encoder(x)
        
        if 'z_np' not in locals():
            z_np = mean.to('cpu').detach().numpy().copy()
        else:
            z_np = np.concatenate([z_np, mean.to('cpu').detach().numpy().copy()])

        for i in range(s[0]):
            if args.color_type == 'activity':
                c.append(sid2act[t[i]])
            elif args.color_type == 'length':
                c.append(len(sid2seq[t[i]].replace('-','')))
            else:
                print(f"Unknown color type ({args.color_type})")
                exit(0)
                
            s_z_np = z_np.shape
    print(s_z_np)
    
    import umap
    if args.non_rep_umap: # 再現性のないumapを作る
        umap_model = umap.UMAP(n_neighbors=args.nei)
    else: # 再現性のあるumapを作る
        umap_model = umap.UMAP(n_neighbors=args.nei, random_state=42) # n_neighborsのデフォルトは15
            
    z_np = umap_model.fit_transform(z_np) # デフォルトはn_neighbors=15
    
    d1 = list(z_np[:,0])
    d2 = list(z_np[:,1])

        #plt.figure(figsize=(5, 4))
    if args.color_type == 'activity':
        sc = plt.scatter(d1, d2, c=c, s=args.plot_size)
        cbar = plt.colorbar(sc) # このような簡易的な書き方が用意されている。
        cbar.set_label('RNA activity')
    elif args.color_type == 'length' :
        sc = plt.scatter(d1, d2, c=c, s=args.plot_size, cmap='coolwarm')
        cbar = plt.colorbar(sc) # このような簡易的な書き方が用意されている。
        cbar.set_label('RNA length')

    #if args.color_type == 'activity':
    #    cbar.set_label('RNA activity')
    #elif args.color_type == 'length':
    else:
        print(f"Unknown color type ({args.color_type})")
        exit(0)
        
    plt.savefig(png_fname)
    #plt.savefig(f"high_res_{png_fname}", dpi=300)
    plt.close()
        
    return umap_model

def checkArgs(args): # プログラムごとにオプションが異なるため、プログラムごとに設定する、

    if args.act_fname == 'None' and args.color_type == 'activity': # activityファイルが指定されているときだけ、color_typeにactivityが使える
        print("activity file must be provided when using color_type is activity")
        exit(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input sequence and ss file')
    parser.add_argument('model', help='model file name')
    parser.add_argument('color_type', choices=['length', 'activity'], help='type of color, either activity or length')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    #parser.add_argument('--out_dir', default='./', help='output directory')
    #parser.add_argument('--png_prefix', default='', help='prefix of png files')
    #parser.add_argument('--nuc_only', action='store_true', help='nucleotide only model')
    #parser.add_argument('--color_seed', default=42, help='random seed for color')
    parser.add_argument('--non_rep_umap', action='store_true', help='make a non-reproducible UMAP') # 
    #parser.add_argument('--model_type', default='org', choices=['org', 'act'], help='type of the model, either \"org\" or \"act\"')
    parser.add_argument('--act_fname', help='activity file name')
    parser.add_argument('--nei', type=int, default=15, help='number of UMAP neighbors')
    parser.add_argument('--image', default="latent.jpg", help='output image file name')
    parser.add_argument('--plot_size', type=int, default=5, help='size of point')
    parser.add_argument('--save_umap', help='file name of umap model to be saved')
    
    args = parser.parse_args()

    main(args)


