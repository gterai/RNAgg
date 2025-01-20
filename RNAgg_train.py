import sys
import os
import argparse
import copy
import numpy as np
import RNAgg_VAE
import SS2shape3
import matplotlib.pyplot as plt
#import seaborn as sns

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.setrecursionlimit(3000) # 配列が長いと再起の上限に達してしまうので、3000にしている。

NUC_LETTERS = list('ACGU-x')
G_DIM = 11 # 文法を格納する部分の次元

def getGramMatrix(rule: list, max_len:int, dim:int):
    # rule_type 
    # 1(ss): S->aS 
    # 2(st): S->T  
    # 3(tt): T->Ta
    # 4(tu): T->U
    # 5(tb): T->TU
    # 6(us): U->aSb
    g_mat = np.zeros((max_len, dim))

    for r in rule: # ここを間違えているとメチャクチャなことになるので注意すること。
        r_type = r[0] # ルールタイプ
        if r_type == 1: # S->aS
            i, j = r[1]
            g_mat[i,0] = 1. # 既存のプログラムを再度動かせるように、上に積んである。
            g_mat[j,1] = 1. # 既存のプログラムを再度動かせるように、上に積んである。
        elif r_type == 2: # S->T
            i, j = r[1]
            g_mat[i,2] = 1.
            g_mat[j,3] = 1.
        elif r_type == 3: # T->Ta
            i, j = r[1]
            g_mat[i,4] = 1.  # 既存のプログラムを再度動かせるように、上に積んである。
            g_mat[j,5] = 1. # 既存のプログラムを再度動かせるように、上に積んである。
        elif r_type == 4: # T->U
            i, j = r[1]
            g_mat[i,6] = 1.
            g_mat[j,7] = 1.
        elif r_type == 5: # T->TU
            i, j = r[1]
            g_mat[i,8] = 1.
            g_mat[j,9] = 1.
            # branching point
            k = r[2]
            g_mat[k,10] = 1.
            #print(i,k,j)
        elif r_type == 6: # U->aSb
            pass
        else:
            print(f"Unknown rule type")

    """
    for r in rule: # ここを間違えているとメチャクチャなことになるので注意すること。
        r_type = r[0] # ルールタイプ
        if r_type == 1: # S->aS pass
            i, j = r[1]
            g_mat[i,7] = 1. # 既存のプログラムを再度動かせるように、上に積んである。
            g_mat[j,8] = 1. # 既存のプログラムを再度動かせるように、上に積んである。
        elif r_type == 2: # S->T
            i, j = r[1]
            g_mat[i,0] = 1.
            g_mat[j,1] = 1.
        elif r_type == 3: # T->Ta
            i, j = r[1]
            g_mat[i,9] = 1.  # 既存のプログラムを再度動かせるように、上に積んである。
            g_mat[j,10] = 1. # 既存のプログラムを再度動かせるように、上に積んである。
        elif r_type == 4: # T->U
            i, j = r[1]
            g_mat[i,2] = 1.
            g_mat[j,3] = 1.
        elif r_type == 5: # T->TU
            i, j = r[1]
            g_mat[i,4] = 1.
            g_mat[j,5] = 1.
            # branching point
            k = r[2]
            g_mat[k,6] = 1.
            #print(i,k,j)
        elif r_type == 6: # U->aSb
            pass
        else:
            print(f"Unknown rule type")
    """

    return g_mat

def readInput(fname: str):
    d_seq = {}
    d_ss  = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            items = line.split(' ')
            sid, seq, ss = items[0:3]
            d_seq[sid] = seq
            d_ss[sid]  = ss
    return d_seq, d_ss

def get_token2idx(): # Letterと行番号の辞書を作成
    d = {}
    for i,x in enumerate(NUC_LETTERS):
        d[x] = i
    return d

"""
def outputScoreMatrix(score, png_name):
    #s = score.shape
    #if s[0] != s[1]:
    #    print(f"Input matrix is not square ({s[0]}x{s[1]})", file=sys.stderr)
    #    exit(0)
    
    sns.heatmap(score)
    plt.savefig(png_name)
    plt.close()
"""

def main(args: dict):
    
    # オプションのチェック
    # このプログラムの出力は、pthとpngでout_dirに出力するようになっている。
    if '/' in args.png_prefix:
        print(f"args.png_prefix({args.png_prefix}) should not contain '/'.", file=sys.stderr)
        exit(0)
    if not os.path.exists(args.out_dir) or not os.path.isdir(args.out_dir):
        print(f"args.out_dir({args.out_dir}) does not exist.", file=sys.stderr)
        exit(0)

    model_path = args.out_dir + '/' + args.model_fname
    
    token2idx = get_token2idx()
    idx2token = dict([y,x] for x,y in token2idx.items())
    #print(token2idx) # vocablaryのチェック
    #print(idx2token) # vocablaryのチェック
    
    sid2seq, sid2ss  = readInput(args.input)
    max_len = max([len(x) for x in sid2seq.values()])
    print(f"Maximul_length={max_len}")
    
    sid_list = list(sid2seq.keys())
    B_mat = [] # binary_matrix
    
    word_size = len(NUC_LETTERS) 
    VDIM = word_size + G_DIM # バイナリベクトルの次元
    # 20241025ここまでチェックした。
    
    for n, sid in enumerate(sid_list):
        seq, ss = sid2seq[sid], sid2ss[sid]
        
        if '()' in ss:
            print (f"{sid} is excluded because it contains base-bair of neibouring nucleotides.", file=sys.stderr)
            continue

        # padding for unaligned sequences
        l = len(seq)
        dif = max_len - l
        seq += 'x' * dif
        ss  += '.' * dif
        
        #ここでtreeを作成する
        bp = SS2shape3.getBPpos_ij(ss)
        r = []
        SS2shape3.generate_rule_G4b(0, r, seq.lower(), ss, bp, (0,len(ss)-1), 'S') ## ssも入れることにする。その方がわかりやすい

        #print(r)
        #exit(0)
        
        # 文法の行列を作成する。
        g_mat = torch.tensor(getGramMatrix(r, max_len, G_DIM)).float() # 文法のdimensionはG_DIM
        
        # 塩基の行列を作成する。
        nuc_token_list = []
        for i in range(max_len):
            token_id = token2idx[seq[i]]
            nuc_token_list.append(token_id)
        n_mat = F.one_hot(torch.tensor(nuc_token_list), word_size).float()
        #print(n_mat)
        
        if args.nuc_only:
            B_mat.append(n_mat)
        else:
            tmp_mat = torch.cat((n_mat, g_mat), dim=1)
            B_mat.append(tmp_mat)

        # 入力行列の確認
        if(0):
            sns.heatmap(input_mat)
            plt.savefig("input.png")
            plt.close()
            exit(0)
    
    d = Dataset(B_mat, sid_list)
    train_dataloader = DataLoader(d, batch_size=args.s_bat, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, file=sys.stderr)
    
    if args.nuc_only:
        model = RNAgg_VAE.MLP_VAE(max_len*(word_size), max_len*(word_size), args.d_rep, device=device).to(device) 
    else:
        model = RNAgg_VAE.MLP_VAE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), args.d_rep, device=device).to(device) 

    # パラメータカウント
    if(0):
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        print(params)  # 811876
        exit(0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #torch.autograd.set_detect_anomaly(True)
    best_model = ''
    best_loss = 1e100
    best_epoch = 0
    Loss_list = []
    L1_list = []
    L2_list = []
    for epoch in range(args.epoch):
        loss_train = 0
        acc_train = 0
        
        pred = []
        obs = []
        
        loss_mean = 0.
        L1_mean = 0.
        L2_mean = 0.
        CEloss = nn.CrossEntropyLoss()
        sigmoid = nn.Sigmoid()
        BCEloss = nn.BCELoss()
        for x, t in train_dataloader:
            s = x.shape
            #x = x.view(s[0], max_len*(word_size+7)) # ここで一列にする。CNNにするなら、一列にしない。RNAVAE.py側で吸収した方が良い。
            #s = x.shape
            #print(s)
            #exit(0)
            x = x.to(device)
            #rm = rm.to(device)
            #print(s)
            #exit(0)
            model.train()
            
            mean, var = model.encoder(x)
            
            z = model.reparameterize(mean, var)
 
            if(torch.sum(torch.isnan(z)).item()):
                print("Encoder output contains nan", file=sys.stderr)
                print(z)
                exit(0)
 
            y = model.decoder(z)

            if args.nuc_only: # nucleotide only model
                yy = y.view(s[0], max_len, word_size)
                L1 = CEloss(torch.transpose(yy,1,2), torch.transpose(x,1,2))
            else:
                yy = y.view(s[0], max_len, word_size+G_DIM)
                #print(torch.max(yy))
                #print(torch.min(yy))
                #print(torch.max(sigmoid(yy[:,:,6:])))
                #print(torch.min(sigmoid(yy[:,:,6:])))
                #exit(0)
                L1_nuc = CEloss(torch.transpose(yy[:,:,:6],1,2), torch.transpose(x[:,:,:6],1,2))
                L1_ss = BCEloss(sigmoid(yy[:,:,6:]), x[:,:,6:]) 
                L1 = L1_nuc + L1_ss
                #L1 = L1_nuc + 10 * L1_ss

            L2 = - 1/2 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var, dim = 1))
            loss = L1 + args.beta * L2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mean += loss.item()
            L1_mean += L1.item()
            L2_mean += L2.item()

        loss_mean /= len(train_dataloader)
        L1_mean /= len(train_dataloader)
        L2_mean /= len(train_dataloader)
    
        Loss_list.append(loss_mean)
        L1_list.append(L1_mean)
        L2_list.append(L2_mean)
        print('Epoch: {}, loss: {:.3f}, L1: {:.3f}, L2: {:.3f}'.format(epoch+1, loss_mean, L1_mean, L2_mean), file=sys.stderr)

        if loss_mean <= best_loss:
            best_loss = loss_mean
            # コピーに時間がかかるので、普段は使わない。
            best_model = copy.deepcopy(model) # モデルが持つnon-leafなtensorをコピーできないらしい
            best_epoch = epoch+1
            
        if args.save_ongoing > 0 and epoch != 0 and (epoch + 1) % args.save_ongoing == 0:
            # モデルとパラメータの保存
            save_model(best_model, args.d_rep, max_len, best_epoch, args.epoch, args.lr, args.beta, model_path)
            
            # latent spaceの描画(指定されたepoch ごと)
            latent_png_name = args.out_dir + '/' + args.png_prefix + f'latent_{epoch+1}.png'
            draw_latent(latent_png_name, best_model, train_dataloader, sid2seq, device)
            
            
    # モデルの保存
    save_model(best_model, args.d_rep, max_len, best_epoch, args.epoch, args.lr, args.beta, model_path)
    print(f"The best model was obtained at epoch={best_epoch}")

    # lossとlatent spaceの描画
    Loss_png_names = ("Loss.png", "L1.png", "L2.png")
    Loss_png_names = [args.png_prefix + x for x in Loss_png_names]
    Loss_png_names = [args.out_dir + '/' + x for x in Loss_png_names]
    draw_loss(Loss_png_names, (Loss_list, L1_list, L2_list))
    latent_png_name = args.out_dir + '/' + args.png_prefix + 'latent.png'
    draw_latent(latent_png_name, best_model, train_dataloader, sid2seq, device)

def draw_latent(png_fname, best_model, train_dataloader, sid2seq, device):
    best_model.eval()
    
    # zの図示
    if(1):
        if 'z_np' in locals(): # これは関数にしたら不要と思うが残しておく。リリース時には削除する。
            del(z_np)
        d1 = []
        d2 = []
        c = []
        for (x,t) in train_dataloader:
            x = x.to(device)
            s = x.shape
            #x = x.view(s[0], MAX_LEN*(word_size+7)) # ここで一列にする。 CNNにするなら、一列にしない。RNAVAE.py側で吸収した方が良い。
            mean, var = best_model.encoder(x)
            #z = model.encoder(x)
            if 'z_np' not in locals():
                z_np = mean.to('cpu').detach().numpy().copy()
            else:
                z_np = np.concatenate([z_np, mean.to('cpu').detach().numpy().copy()])

            for i in range(s[0]):
                c.append(len(sid2seq[t[i]].replace('-','')))
                        
            s_z_np = z_np.shape
        print(s_z_np)
                
        if s_z_np[1] > 2 and s_z_np[0] > 2: # drepが２より大きい時はUMAPを使う。
            import umap
            #z_np = umap.UMAP(n_neighbors=len(z_np)-1).fit_transform(z_np)
            z_np = umap.UMAP().fit_transform(z_np) # デフォルトはn_neighbors=15
            
        d1 += list(z_np[:,0])
        d2 += list(z_np[:,1])
        sc = plt.scatter(d1, d2, c=c)
        #sc = plt.scatter(d1, d2)
        plt.colorbar(sc) # このような簡易的な書き方が用意されている。
        plt.savefig(png_fname)
        plt.close()

def draw_loss(loss_fname_tuple: tuple, loss_list_tuple: tuple):
    Loss_name, L1_name, L2_name = loss_fname_tuple
    Loss_list, L1_list, L2_list = loss_list_tuple
    
    plt.plot(Loss_list)
    plt.savefig(Loss_name)
    plt.close()

    plt.plot(L1_list)
    plt.savefig(L1_name)
    plt.close()
            
    plt.plot(L2_list)
    plt.savefig(L2_name)
    plt.close()


def save_model(best_model, d_rep, max_len, best_epoch, max_epoch, lr, beta, model_path):
    save_model_dict = {
        'model_state_dict':best_model.state_dict(),
        'd_rep':d_rep,
        'max_len':max_len,
        'best_epoch':best_epoch,
        'max_epoch':max_epoch,
        'lr':lr,
        'beta':beta
    }
    torch.save(save_model_dict, model_path)


class Dataset:
    def __init__(self, input_mat, sid_list): 
        self.data   = input_mat
        self.sid_list = sid_list

    def __getitem__(self, index):
        return self.data[index], self.sid_list[index]
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input sequence and ss file')
    parser.add_argument('--epoch', type=int, default=500, help='maximum epoch')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float,  default=0.001, help='learning rate')
    parser.add_argument('--beta', type=float,  default=0.001, help='hyper parameter beta')
    parser.add_argument('--d_rep', type=int,  default=8, help='dimension of latent vector')
    parser.add_argument('--out_dir', default='./', help='output directory')
    parser.add_argument('--model_fname', default='model_RNAgg.pth', help='model file name')
    parser.add_argument('--png_prefix', default='', help='prefix of png files')
    parser.add_argument('--save_ongoing', default=0, type=int, help='save model and latent spage during training')
    parser.add_argument('--nuc_only', action='store_true', help='nucleotide only model')
    args = parser.parse_args()

    main(args)
