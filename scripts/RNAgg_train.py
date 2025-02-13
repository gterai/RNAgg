import sys
import os
import argparse
import copy
import numpy as np
import RNAgg_VAE
import SS2shape3
import matplotlib.pyplot as plt
import utils_gg as utils
import Binary_matrix

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

    model_path = args.out_dir + '/' + args.model_fname
    
    token2idx = utils.get_token2idx(NUC_LETTERS)
    idx2token = dict([y,x] for x,y in token2idx.items())
    #print(token2idx) # vocablaryのチェック
    #print(idx2token) # vocablaryのチェック
    
    sid2seq, sid2ss  = utils.readInput(args.input)
    max_len = max([len(x) for x in sid2seq.values()])
    print(f"Maximam_length={max_len}")
    
    sid_list = list(sid2seq.keys())
    word_size = len(NUC_LETTERS)  # 塩基の種類
    VDIM = word_size + G_DIM # バイナリベクトルの次元

    # args.nuc_onlyが何度も出てくる。
    # 1. args.nuc_onlyがRNAgg_trainでしか使わない
    # 2. 他のプログラムではモデルから得たnuc_yes_noを用いる
    # 3. それらの中間に位置するプログラムが存在する
    # 以上３点から、このような実装になるのは仕方ない。
    if args.nuc_only: 
        nuc_yes_no = 'yes'
    else:
        nuc_yes_no = 'no'
        
    B_mat = Binary_matrix.makeMatrix(sid2seq, sid2ss, sid_list, word_size, G_DIM, token2idx, nuc_yes_no)
    
    if args.act_fname == None:  # activity fileが指定されていないときはダミーを作る
        sid2act = dict([(sid, np.nan) for sid in sid_list])
    else:
        sid2act = utils.readAct(args.act_fname)
    
    act_list = [sid2act[sid] for sid in sid_list]
    d = utils.Dataset(B_mat, sid_list, act_list)
    train_dataloader = DataLoader(d, batch_size=args.s_bat, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", file=sys.stderr)

    if args.act_fname == None: # activity fileが指定されていない -> org model
        if args.nuc_only:
            model = RNAgg_VAE.MLP_VAE(max_len*(word_size), max_len*(word_size), args.d_rep, device=device).to(device) 
        else:
            model = RNAgg_VAE.MLP_VAE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), args.d_rep, device=device).to(device) 
    else: # activity fileが指定されている -> act model
        if args.nuc_only:
            model = RNAgg_VAE.MLP_VAE_REGRE(max_len*(word_size), max_len*(word_size), args.d_rep, device=device).to(device) 
        else:
            model = RNAgg_VAE.MLP_VAE_REGRE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), args.d_rep, device=device).to(device)
 
           
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
    L3_list = [] # MSE loss for activity
    for epoch in range(args.epoch):
        loss_train = 0
        acc_train = 0
        
        pred = []
        obs = []
        
        loss_mean = 0.
        L1_mean = 0.
        L2_mean = 0.
        L3_mean = 0.
        CEloss = nn.CrossEntropyLoss(reduction="none")
        #CEloss_mean = nn.CrossEntropyLoss()
        sigmoid = nn.Sigmoid()
        BCEloss = nn.BCELoss(reduction="none")
        #BCEloss_mean = nn.BCELoss()
        MSEloss = nn.MSELoss()
        for x, t, v in train_dataloader:
            s = x.shape
            bs, L = s[0], s[1]
            #x = x.view(s[0], max_len*(word_size+7)) # ここで一列にする。CNNにするなら、一列にしない。RNAVAE.py側で吸収した方が良い。
            #s = x.shape
            x = x.to(device)
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
                L1 = torch.mean(torch.sum(CEloss(torch.transpose(yy,1,2), torch.transpose(x,1,2)), dim=1)/L) # logなので、確率の掛け算はsumになる。RNAの長さで割っておく。
            else:
                yy = y.view(s[0], max_len, word_size+G_DIM)
                L1_nuc_sum = torch.sum(CEloss(torch.transpose(yy[:,:,:6],1,2), torch.transpose(x[:,:,:6],1,2)), dim=1)/L  # logなので、確率の掛け算はsumになる。RNAの長さで割っておく。
                L1_ss_sum = torch.sum(torch.sum(BCEloss(sigmoid(yy[:,:,6:]), x[:,:,6:]), dim=2), dim=1)/(G_DIM * L)  # logなので、確率の掛け算はsumになる。行列の要素数で割っておく。
                L1_sum = L1_nuc_sum + L1_ss_sum
                L1 = torch.mean(L1_sum)

                #tmp_L1_nuc = CEloss_mean(torch.transpose(yy[:,:,:6],1,2), torch.transpose(x[:,:,:6],1,2)) # 単にmeanを撮るのと同じになる、
                #tmp_L1_ss = BCEloss_mean(sigmoid(yy[:,:,6:]), x[:,:,6:]) # 単にmeanを撮るのと同じになる、
                #tmp_L1 = tmp_L1_nuc + tmp_L1_ss
                #print(L1, tmp_L1)
                
            L2 = - 1/2 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var, dim = 1))
            loss_vae = L1 + args.beta * L2
            
            if args.act_fname == None: # org model
                loss = loss_vae
            else: # act model
                v = v.to(device)
                pred_act = model.regress(mean)
                loss_reg = MSEloss(pred_act, v.view(s[0],1)) 
                loss = loss_reg + loss_vae
                L3_mean += loss_reg.item()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mean += loss.item()
            L1_mean += L1.item()
            L2_mean += L2.item()
            
        loss_mean /= len(train_dataloader)
        L1_mean /= len(train_dataloader)
        L2_mean /= len(train_dataloader)
        L3_mean /= len(train_dataloader)
        
        Loss_list.append(loss_mean)
        L1_list.append(L1_mean)
        L2_list.append(L2_mean)
        L3_list.append(L3_mean)


        if args.act_fname == None: # org model
            print('Epoch: {}, loss: {:.3f}, L1: {:.3f}, L2: {:.3f}'.format(epoch+1, loss_mean, L1_mean, L2_mean), file=sys.stderr)
        else: # act model
            print('Epoch: {}, loss: {:.3f}, L1: {:.3f}, L2: {:.3f}, L3: {:.3f}'.format(epoch+1, loss_mean, L1_mean, L2_mean, L3_mean), file=sys.stderr)

        
        if loss_mean <= best_loss:
            best_loss = loss_mean
            # コピーに時間がかかるので、普段は使わない。
            best_model = copy.deepcopy(model) # モデルが持つnon-leafなtensorをコピーできないらしい
            best_epoch = epoch+1
            
        if args.save_ongoing > 0 and epoch != 0 and (epoch + 1) % args.save_ongoing == 0:
            # モデルとパラメータの保存
            #save_model(best_model, args.d_rep, max_len, best_epoch, args.epoch, args.lr, args.beta, model_path, args)
            save_model(best_model, max_len, best_epoch, model_path, args)
            
            # latent spaceの描画(指定されたepoch ごと)
            latent_png_name = args.out_dir + '/' + args.png_prefix + f'latent_{epoch+1}.png'
            #draw_latent(latent_png_name, best_model, train_dataloader, sid2seq, device, args)
            
            
    # モデルの保存
    save_model(best_model, max_len, best_epoch, model_path, args)
    #save_model(best_model, args.d_rep, max_len, best_epoch, args.epoch, args.lr, args.beta, model_path)
    #save_model(best_model, args.d_rep, max_len, best_epoch, args.epoch, args.lr, args.beta, model_path)
    print(f"The best model was obtained at epoch={best_epoch}")

    # lossとlatent spaceの描画
    Loss_png_names = ("Loss.png", "L1.png", "L2.png")
    Loss_png_names = [args.png_prefix + x for x in Loss_png_names]
    Loss_png_names = [args.out_dir + '/' + x for x in Loss_png_names]
    draw_loss(Loss_png_names, (Loss_list, L1_list, L2_list))
    latent_png_name = args.out_dir + '/' + args.png_prefix + 'latent.png'
    #draw_latent(latent_png_name, best_model, train_dataloader, sid2seq, device, args)


def draw_loss(loss_fname_tuple: tuple, loss_list_tuple: tuple):

    for fname, loss_list in zip(loss_fname_tuple, loss_list_tuple):
        
        plt.plot(loss_list)
        plt.savefig(fname)
        plt.close()

#def save_model(best_model, d_rep, max_len, best_epoch, max_epoch, lr, beta, model_path, args):
def save_model(best_model, max_len, best_epoch, model_path, args):

    if args.act_fname == None:
        model_type = 'org'
    else:
        model_type = 'act'

    if args.nuc_only:
        nuc_yes_no = 'yes'
    else:
        nuc_yes_no = 'no'

    save_model_dict = {
        'model_state_dict':best_model.state_dict(),
        'd_rep':args.d_rep,
        'max_len':max_len,
        'best_epoch':best_epoch,
        'max_epoch':args.epoch,
        'lr':args.lr,
        'beta':args.beta,
        'type': model_type,
        'nuc_only': nuc_yes_no
    }
    torch.save(save_model_dict, model_path)

def checkArgs(args): # プログラムごとにオプションが異なるため、プログラムごとに設定する、

    if '/' in args.png_prefix:
            print(f"args.png_prefix({args.png_prefix}) should not contain '/'.", file=sys.stderr)
            exit(0)
                
    if not os.path.exists(args.out_dir) or not os.path.isdir(args.out_dir):
        print(f"args.out_dir({args.out_dir}) does not exist.", file=sys.stderr)
        exit(0)


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
    #parser.add_argument('--model_type', default='org', choices=['org', 'act'], help='type of the model, either \"org\" or \"act\"')
    parser.add_argument('--act_fname', help='activity file name')
    args = parser.parse_args()

    main(args)
