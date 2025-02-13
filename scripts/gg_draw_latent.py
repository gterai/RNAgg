import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils_gg as utils
import random
import pickle

#from torch.utils.data import DataLoader
#
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import copy
#import RNAgg_VAE
#import SS2shape3


#NUC_LETTERS = list('ACGU-x')
#G_DIM = 11 # 文法を格納する部分の次元

def main(args: dict):
    
    # embedding (pickle形式）の読み込み
    f = open(args.emb_inf, 'rb')
    emb_inf = pickle.load(f)
    f.close()
    
    sid_list = emb_inf['sid_list']
    z_np = emb_inf['emb']
        
    if args.color_fname != None: # activity dataが指定されていれば得る
        sid2color = utils.readAct(args.color_fname)
    else: # 指定されていない場合には、ダミー値を入れる
        sid2color = dict([(sid, np.nan) for sid in sid_list])
        
    color_list = [sid2color[sid] for sid in sid_list]
    
    umap_model = draw_latent_from_emb(args.image, z_np, color_list, args)
    
    # umapモデルの保存
    if args.save_umap:
        with open(args.save_umap, "wb") as f:
            pickle.dump(umap_model, f)
            
def draw_latent_from_emb(png_fname, z_np, color_list, args):
    
    import umap
    if args.rep_umap: # 再現性のあるumapを作る
        umap_model = umap.UMAP(n_neighbors=args.nei, random_state=42) # n_neighborsのデフォルトは15
    else: # 再現性のないumapを作る
        umap_model = umap.UMAP(n_neighbors=args.nei)
        
    z_np = umap_model.fit_transform(z_np) # デフォルトはn_neighbors=15
    
    if (0): # テスト用に200データで描画
        z_np = z_np[0:200] 
        color_list = color_list[0:200] 
        print("This is a test using 200 data points", file=sys.stderr)

    d1 = list(z_np[:,0])
    d2 = list(z_np[:,1])
    
    # d1, d2, cをランダムシャフルする。
    random.seed(42)
    index_list = [i for i in range(len(color_list))]
    random.shuffle(index_list)
    d1 = [d1[i] for i in index_list]
    d2 = [d2[i] for i in index_list]
    color_list  = [color_list[i]  for i in index_list]

    fsize = [float(x) for x in args.fig_size.split(',')]
    fig, ax = plt.subplots(figsize=fsize)
    if args.color_fname:
        sc = plt.scatter(d1, d2, c=color_list, s=args.plot_size)
        if not args.no_color_scale: # カラースケールを表示
            cbar = plt.colorbar(sc) # このような簡易的な書き方が用意されている。
            if args.color_label:
                cbar.set_label(args.color_label, fontsize=14)
    else:
        sc = plt.scatter(d1, d2, s=args.plot_size)
    
    ax.set_xlabel('umap-1', fontsize=14)
    ax.set_ylabel('umap-2', fontsize=14)
    
    # 四角形を書き込む    
    if args.points:
        from matplotlib.patches import Rectangle
        points = readPoints(args.points)
        for p_inf in points:
            left, right = sortPoints(p_inf[0], p_inf[1]) # leftは左下、rightは右下の点
            name  = p_inf[2]
            print(left, right)
            hight = right[1] - left[1]
            width = right[0] - left[0]
            rect = Rectangle((left), width, hight, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(left[0], right[1], name, fontsize=24, color='red', ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(png_fname, dpi=args.dpi)
    #plt.savefig(f"high_res_{png_fname}", dpi=300)
    plt.close()

    return umap_model

def readPoints(fname:str):
    l = []
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            p1, p2, name = line.split()
            l.append((p1, p2, name))
    return l

def sortPoints(p1:str, p2:str):
    p1 = [float(v) for v in p1.split(",")]
    p2 = [float(v) for v in p2.split(",")]
    if p1[0] < p2[0]:
        x_min, x_max = p1[0], p2[0]
    else:
        x_min, x_max = p2[0], p1[0]
    if p1[1] < p2[1]:
        y_min, y_max = p1[1], p2[1]
    else:
        y_min, y_max = p2[1], p1[1]

    return (x_min, y_min), (x_max, y_max)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_inf', help='embedding pickle file name')
    parser.add_argument('--color_fname', help='a file containing color information')
    parser.add_argument('--color_label', help='label associated with color scale')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    parser.add_argument('--rep_umap', action='store_true', help='make a non-reproducible UMAP') # 
    parser.add_argument('--nei', type=int, default=15, help='number of UMAP neighbors')
    parser.add_argument('--image', default="latent.jpg", help='output image file name')
    parser.add_argument('--plot_size', type=int, default=5, help='size of point')
    parser.add_argument('--save_umap', help='file name of umap model to be saved')
    parser.add_argument('--points', help='points cordinate data file')
    parser.add_argument('--fig_size', default='6.4,4.8', help='size of figure')
    parser.add_argument('--dpi', type=int, default=100, help='dpi')
    parser.add_argument('--no_color_scale', action='store_true', help='do not show the color scale')
    #parser.add_argument('input', help='input sequence and ss file')
    #parser.add_argument('model', help='model file name')
    #parser.add_argument('color_type', choices=['length', 'activity'], help='type of color, either activity or length')
    #parser.add_argument('--out_dir', default='./', help='output directory')
    #parser.add_argument('--png_prefix', default='', help='prefix of png files')
    #parser.add_argument('--nuc_only', action='store_true', help='nucleotide only model')
    #parser.add_argument('--color_seed', default=42, help='random seed for color')
    #parser.add_argument('--model_type', default='org', choices=['org', 'act'], help='type of the model, either \"org\" or \"act\"')
    #parser.add_argument('--act_fname', help='activity file name')
    
    args = parser.parse_args()

    main(args)


