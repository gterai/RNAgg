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
import utils_gg as utils
#import SS2shape3
#import graphviz
from torch.utils.data import DataLoader
#from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from joblib import Parallel, delayed

torch.set_grad_enabled(False) # generationの時はgradientの計算は行わない。

NUC_LETTERS = list('ACGU-x')
#SS_LETTERS = list('.()')
#S_BAT = 10
#MAX_LEN=93

G_DIM=11

#def get_token2idx():
#    d = {}
#    for i,x in enumerate(NUC_LETTERS):
#        d[x] = i
#    return d

def main(args: dict):
    
    token2idx = utils.get_token2idx(NUC_LETTERS)
    idx2token = dict([y,x] for x,y in token2idx.items())
    word_size = len(NUC_LETTERS)
    #print(word_size)
    #exit(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, file=sys.stderr)
    
    # モデルの初期化とロード
    #checkpoint = torch.load(args.model, weights_only=True)
    checkpoint = torch.load(args.model, map_location=torch.device(device))
    #d_rep, max_len = checkpoint['d_rep'], checkpoint['max_len']
    d_rep, max_len, model_type, nuc_only = checkpoint['d_rep'], checkpoint['max_len'], checkpoint['type'], checkpoint['nuc_only']
    print(f"model type:", model_type, file=sys.stderr)
    print(f"nuc_only:", nuc_only, file=sys.stderr)

    if model_type == 'act':
        if nuc_only == 'yes':
            model = RNAgg_VAE.MLP_VAE_REGRE(max_len*(word_size), max_len*(word_size), d_rep, device=device).to(device) 
        else:
            model = RNAgg_VAE.MLP_VAE_REGRE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), d_rep, device=device).to(device)     
    else:
        if nuc_only == 'yes':
            model = RNAgg_VAE.MLP_VAE(max_len*(word_size), max_len*(word_size), d_rep, device=device).to(device) 
        else:
            model = RNAgg_VAE.MLP_VAE(max_len*(word_size+G_DIM), max_len*(word_size+G_DIM), d_rep, device=device).to(device) 
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    if args.from_emb: # embeddingを直で入力する。
        with open(args.n, "rb") as file:
            z = pickle.load(file)
            z = z.to(device)
    else:
        z = torch.normal(mean=0, std=1, size=(int(args.n), d_rep))
        z = z.to(device)
        
    s_z = z.shape
    y = model.decoder(z)
    
    Sigmoid = nn.Sigmoid()
    Softmax = nn.Softmax(dim=2)
    if nuc_only == 'yes':
        yy = y.view(s_z[0], max_len, word_size)
        yy = Softmax(yy) 
    else:
        yy = y.view(s_z[0], max_len, word_size+G_DIM) # yyはlogitなので、0-1に正規化する必要がある。
        #print(torch.max(yy[:,:,6:]))
        #print(torch.min(yy[:,:,6:]))
        #print(torch.max(yy[:,:,:6]))
        #print(torch.min(yy[:,:,:6]))
        #exit(0)
        yy[:,:,:6] = Softmax(yy[:,:,:6])  # nucleotideの部分は、Softmaxをとる
        yy[:,:,6:] = Sigmoid(yy[:,:,6:])  # grammarの部分は、Sigmoidをとる
 
    gen_seq_list  = []
    gen_score_list = []
    gen_ss_list = []
    if nuc_only == 'yes':
        max_p, max_idx   = torch.max(yy, dim=2)

        max_logP = torch.log(max_p)
        for i in range(len(max_idx)):
            logP = torch.sum(max_logP[i])
            nuc_seq = "".join([idx2token[idx.item()] for idx in max_idx[i]])
            print(i, logP.item(), nuc_seq)
            gen_seq_list.append(trimX(nuc_seq))
            gen_score_list.append(logP.item())
            gen_ss_list.append('') # 空文字を入れる
            
    else: # RNAggモデル
        s = yy.shape
        #print(s)
        #print(type(yy))
        gen_seq_list = []
        gen_ss_list = []
        gen_score_list = []
        for i in range(0, s[0], args.s_bat):
            #print(i, i + args.s_bat)
            yy_batch = yy[i:i + args.s_bat] # スライスの終わりがyyのサイズを超えても、yyの最大サイズになる。
            #print(yy_batch.shape)
            
            seq_list, ss_list, score_list = generateRNA(yy_batch, idx2token, args)
            #generateRNA(yy_batch, None, device, None, None, token2idx, idx2token, args)

            gen_seq_list += seq_list
            gen_ss_list += ss_list
            gen_score_list += score_list
            
    # 最終出力
    fout = open(args.outfile, 'w')
    if args.out_fasta: # fastaフォーマット
        for i in range(len(gen_seq_list)): 
            #print (f">gen{i} {gen_score_list[i]}", file=fout)
            print (f">gen{i}", file=fout)
            out_seq = gen_seq_list[i].replace('-', '')
            print (out_seq, file=fout)
    else: # デフォルトの出力
        for i in range(len(gen_seq_list)): 
            out_seq = gen_seq_list[i].replace('-', '')
            out_ss  = gen_ss_list[i].replace('-', '')
            print (f"gen{i}", out_seq, out_ss, file=fout) # 2次構造は出力しない
    fout.close()
    

#def generateOneSeq(n, pssm_batch_log, StoS, StoT, TtoT, TtoU, TtoTU, pair_numera, sid2seq, sid2ss, idx2token):
def generateOneSeq(n, pssm_batch_log, StoS, StoT, TtoT, TtoU, TtoTU, pair_numera, idx2token):
    s = pssm_batch_log.shape
    p_type2pair = {1:'AU',2:'UA',3:'GC',4:'CG',5:'GU',6:'UG'}
    
    I = {'S':0, 'T':1, 'U':2}
    MIN_LOOP = 3
    
    sid = None
        
    if args.outProb:
        if (n > 10):
            print(f"Generation of score. matrix for {n} is skipped.", file=sys.stderr)
        else:
            outputScoreMatrix(TtoU[n],  f"scoreTtoU_{n}.png")
            outputScoreMatrix(TtoTU[n], f"scoreTtoTU_{n}.png")
            outputScoreMatrix(StoT[n],  f"scoreStoT_{n}.png")
            
    pssm_log = pssm_batch_log[n]
    
    k_set = set() # bifucationの可能性が1e-5のポジションのみを考える。
    for k in range(len(pssm_log[:,16])):
        if pssm_log[k,16] >= np.log(1e-5):
            k_set.add(k)
            
    S = np.full((s[1],s[1]), np.nan) # L x Lの行列
    T = np.full((s[1],s[1]), np.nan) # L x Lの行列
    U = np.full((s[1],s[1]), np.nan) # L x Lの行列

    # トレースバック用の行列
    TR  = [[[(np.nan,np.nan,np.nan) for j in range(s[1])] for i in range(s[1])] for st in range(3)] # 3 x L x Lの行列（Pathを記録する）
    #print(TR)
    TB = [[[(np.nan,np.nan,np.nan) for j in range(s[1])] for i in range(s[1])] for st in range(3)] # 3 x L x Lの行列（Pathを記録する）
    
    #assert_flg = 0

    for i in range(s[1]): # 長さ1
        S[i,i] = max(pssm_log[i,:6]) # ACGU-x
        max_base = idx2token[np.argmax(pssm_log[i,:6])] 
        #print(i, max_base)
        #print(max_base)
        TR[I['S']][i][i] = (I['S'],i+1,i)
        TB[I['S']][i][i] = max_base
        T[i,i] = -1e100
        U[i,i] = -1e100
    for l in range(2, MIN_LOOP+2): # 長さ2から４まで
        for i in range(s[1]-l+1):
            j = i + l - 1
            S[i,j] = max(pssm_log[i,:6]) + StoS[n,i,j] + S[i+1,j] # StoSが文法の部分、maxが塩基の部分
            TR[I['S']][i][j] = (I['S'],i+1,j)
            max_base = idx2token[np.argmax(pssm_log[i,:6])]
            TB[I['S']][i][j] = max_base
            T[i,j] = -1e100 # pairを組めないので、score -1e100
            U[i,j] = -1e100 # pairを組めないので、score -1e100
            
    # DP行列を埋める
    for l in range(MIN_LOOP+2, s[1]+1): # 長さ5からLまで
        for i in range(s[1]-l+1):
            j = i + l - 1
            #print(l, i,j)
            
            # state U
            ## rule 6: U->aSb
            ##6つのpairを評価する。
            U[i,j] = -1e100
            for p_type in range(1,7):
            #for p_type in [1,2,5,6,4,3]:
                score = pair_numera[p_type][n,i,j] + S[i+1,j-1]
                #assert(0 <= pair_numera[p_type] <= 1)
                if U[i][j] < score:
                    U[i][j] = score
                    TR[I['U']][i][j] = (I['S'],i+1,j-1)
                    TB[I['U']][i][j] = p_type2pair[p_type]
            #print(i,j,U[i,j])
            
            # state T
            ## rule 3: T->Ta
            T[i,j] = max(pssm_log[j,:6]) + TtoT[n,i,j] + T[i,j-1]
            
            TR[I['T']][i][j] = (I['T'],i,j-1)
            max_base = idx2token[np.argmax(pssm_log[j,:6])]
            TB[I['T']][i][j] = max_base
            ## rule 4: T->U
            score = TtoU[n,i,j] + U[i,j]
            #if assert_flg == 1: assert(0 <= TtoU[n,i,j]/Zt[n,i,j] <= 1)
            if T[i][j] < score:
                T[i][j] = score
                TR[I['T']][i][j] = (I['U'],i,j)
                TB[I['T']][i][j] = ''
            #print(prob)
            ## rule 5: T->TU
            #bif = 0
            if l >= 10: # 長さ10以上の時のみBifucationを考えることとする。
                #local_max_prob = 0 # 少し早いコードの検証用
                if T[i,j] < TtoTU[n,i,j] and TtoTU[n,i,j] > np.log(1e-5): # ある程度確率が高い場合しか考えない。
                    # T[i,j]よりスコアが低ければ、考える必要がない。
                    #print(i,j)
                    for k in k_set:
                        if k < i+1 or j-1 < k:
                            continue
                        score = TtoTU[n,i,j] + pssm_log[k,16] + T[i,k] + U[k+1,j] # S->TU遷移の確率
                        if T[i][j] < score:
                            #bif = 1
                            T[i][j] = score
                            TR[I['T']][i][j] = [(I['T'],i,k), (I['U'],k+1,j)] # listかtupleかでbifucationかどうかを区別することにする。
                            TB[I['T']][i][j] = ''
                        #exit(0)
            # state S
            ## rule 1: S->aS
            #S[i,j] = max(pssm_log[i,:6]) + StoS[n,i,j] + S[i+1,j] # Sからxの出力を許可
            S[i,j] = max(pssm_log[i,:5]) + StoS[n,i,j] + S[i+1,j]  # Sからxは出力されない
            #if assert_flg == 1:assert(0 <= max(pssm_exp[i,base_idx]/Zs[n,i,j]) <= 1)
            TR[I['S']][i][j] = (I['S'],i+1,j)
            #max_base = idx2token[np.argmax(pssm_log[i,:6])] # Sからxの出力を許可
            max_base = idx2token[np.argmax(pssm_log[i,:5])]  # Sからxは出力されない
            TB[I['S']][i][j] = max_base
            
            ## rule 2: S->T
            score = StoT[n,i,j] + T[i,j]
            #print(i,j, S[i,j], score)
            #if assert_flg == 1:assert(0 <= StoT[n,i,j]/Zs[n,i,j] <= 1)
            if S[i][j] < score:
                S[i][j] = score
                TR[I['S']][i][j] = (I['T'],i,j)
                TB[I['S']][i][j] = ''

    # traeback
    seq = ['*' for i in range(s[1])]
    ss  = ['*' for i in range(s[1])]
    TraceBack('S', (0, s[1]-1), TR, TB, seq, ss)
    gen_seq   = ''.join(seq)
    gen_ss    = ''.join(ss)
    gen_score = S[0,s[1]-1] 
    print("score=", gen_score, file=sys.stderr)
    return gen_seq, gen_ss, gen_score
    
#def generateRNA(pssm_batch, sid_list, device, sid2seq, sid2ss, token2idx, idx2token, args):
def generateRNA(pssm_batch, idx2token, args):

    pssm_batch = torch.clip(pssm_batch, min=1e-10, max=1)
    pssm_batch_log = torch.log(pssm_batch)
    s = pssm_batch.shape

    #base_idx = [0,3,6,9,12]
    #no_base_idx = [1,2,4,5,7,8,10,11]

    # ブロードキャストを利用して、全i,j塩基対のスコアを求める。
    #A:0, C:1, G:2, U:3
    mAU = pssm_batch_log[:,:,0].view(s[0],1,s[1]) + pssm_batch_log[:,:,3].view(s[0],s[1],1) # AU
    mUA = pssm_batch_log[:,:,3].view(s[0],1,s[1]) + pssm_batch_log[:,:,0].view(s[0],s[1],1) # UA
    mGC = pssm_batch_log[:,:,2].view(s[0],1,s[1]) + pssm_batch_log[:,:,1].view(s[0],s[1],1) # GC
    mCG = pssm_batch_log[:,:,1].view(s[0],1,s[1]) + pssm_batch_log[:,:,2].view(s[0],s[1],1) # CG
    mGU = pssm_batch_log[:,:,2].view(s[0],1,s[1]) + pssm_batch_log[:,:,3].view(s[0],s[1],1) # GU
    mUG = pssm_batch_log[:,:,3].view(s[0],1,s[1]) + pssm_batch_log[:,:,2].view(s[0],s[1],1) # UG

    mAU = torch.transpose(mAU, 1, 2) 
    mUA = torch.transpose(mUA, 1, 2) 
    mGC = torch.transpose(mGC, 1, 2) 
    mCG = torch.transpose(mCG, 1, 2) 
    mGU = torch.transpose(mGU, 1, 2)
    mUG = torch.transpose(mUG, 1, 2)

    # 全i,jに対するS->aS遷移
    StoS = pssm_batch_log[:,:,6].view(s[0],1,s[1]) + pssm_batch_log[:,:,7].view(s[0],s[1],1) # StoT
    StoS = torch.transpose(StoS, 1, 2)

    # ブロードキャストを利用して、全i,jに対するS->T遷移と、T->U遷移のスコアを求める。
    StoT = pssm_batch_log[:,:,8].view(s[0],1,s[1]) + pssm_batch_log[:,:,9].view(s[0],s[1],1) # StoT
    StoT = torch.transpose(StoT, 1, 2)

    # 全i,jに対するT->Ta遷移
    TtoT = pssm_batch_log[:,:,10].view(s[0],1,s[1]) + pssm_batch_log[:,:,11].view(s[0],s[1],1) # StoT
    TtoT = torch.transpose(TtoT, 1, 2)

    TtoU = pssm_batch_log[:,:,12].view(s[0],1,s[1]) + pssm_batch_log[:,:,13].view(s[0],s[1],1) # TtoU
    TtoU = torch.transpose(TtoU, 1, 2)

    # ブロードキャストを利用して、全i,jに対するT->TUの遷移スコアを求める。
    TtoTU = pssm_batch_log[:,:,14].view(s[0],1,s[1]) + pssm_batch_log[:,:,15].view(s[0],s[1],1) # TtoTU
    TtoTU = torch.transpose(TtoTU, 1, 2) # tranposeしなくて良いかも。broad castを変えれば良さそう。

    # 次元を分けたことにより、transposeが必要になった。
    
    pair_numera = { 1:mAU.to('cpu').detach().numpy().copy(),
                    2:mUA.to('cpu').detach().numpy().copy(),
                    3:mGC.to('cpu').detach().numpy().copy(),
                    4:mCG.to('cpu').detach().numpy().copy(),
                    5:mGU.to('cpu').detach().numpy().copy(),
                    6:mUG.to('cpu').detach().numpy().copy()}
    #p_type2pair = {1:'AU',2:'UA',3:'GC',4:'CG',5:'GU',6:'UG'}

    # この先に利用するスコア行列をnumpyにする。
    pssm_batch = pssm_batch.to('cpu').detach().numpy().copy()
    pssm_batch_log = pssm_batch_log.to('cpu').detach().numpy().copy()
    
    StoS  = StoS.to('cpu').detach().numpy().copy()
    StoT  = StoT.to('cpu').detach().numpy().copy()
    TtoT  = TtoT.to('cpu').detach().numpy().copy()
    TtoU  = TtoU.to('cpu').detach().numpy().copy()
    TtoTU = TtoTU.to('cpu').detach().numpy().copy()
    
    s = pssm_batch_log.shape # numpyにしておく。
    
    #tasks = [delayed(generateOneSeq)(n, pssm_batch_log, StoS, StoT, TtoT, TtoU, TtoTU, pair_numera, sid2seq, sid2ss, idx2token) for n in range(s[0])]
    tasks = [delayed(generateOneSeq)(n, pssm_batch_log, StoS, StoT, TtoT, TtoU, TtoTU, pair_numera, idx2token) for n in range(s[0])]

    results = Parallel(n_jobs=args.n_cpu)(tasks)
    #gen_seq_list   = [trimX(x[0]) for x in results]
    #gen_ss_list    = [x[1] for x in results]
    tmp_list = [(trimX_SS(x[0], x[1])) for x in results] # 2時構造もトリムする。
    gen_seq_list   = [x[0] for x in tmp_list]
    gen_ss_list    = [x[1] for x in tmp_list]
    gen_score_list = [x[2] for x in results]

    # calc pair frequency
    if(0):
        num_pairs = {}
        for i in range(len(gen_seq_list)):
            bp_list = getBP(gen_ss_list[i])
            for bp in bp_list:
                if bp[1] > len(gen_seq_list[i]): # xをtrimしているので、このようなことが起こりうる。
                    continue
                pair = gen_seq_list[i][bp[0]] + gen_seq_list[i][bp[1]]
                #print(pair)
                if pair not in num_pairs:
                    num_pairs[pair] = 0 # 動的初期化
                num_pairs[pair] += 1
        
        ordered = sorted(num_pairs.items(), key=lambda x:x[1], reverse=True)
        sum_pairs = np.sum(list(num_pairs.values()))

        for pair,num in ordered:
            print(pair, num/sum_pairs, file=sys.stderr)
        #exit(0)

    return gen_seq_list, gen_ss_list, gen_score_list

def getBP(ss:str):
    bp_list = []
    stack = []
    for i in range(len(ss)):
        if ss[i] == '(' or ss[i] == '<':
            stack.append(i)
        elif ss[i] == ')' or ss[i] == '>':
            left = stack.pop()
            bp_list.append((left, i))
        elif ss[i] == '.' or ss[i] == ',' or ss[i] == ':' or \
             ss[i] == '_':
            pass
        else:
            print(f"Unexpected SS annotation ({ss[i]})", file=sys.stderr)
            exit(0)

    return bp_list

def trimX(seq:str):
    s = ''
    for n in seq:
        if n == 'x':
            break
        else:
            s = s + n
    return s

def trimX_SS(seq:str, sec:str):
    s = ''
    ss = ''
    for n, c in zip(seq, sec):
        if n == 'x':
            break
        else:
            s = s + n
            ss = ss + c
    return s, ss


def TraceBack(st_ini, pos_ini, TR, TB, seq, ss):
    ST2IDX = {'S':0, 'T':1, 'U':2}
    IDX2ST = {0:'S', 1:'T', 2:'U'}
    i,j = pos_ini
    stack = [(st_ini,i,j)]

    while(len(stack) > 0):
        (st, i, j) = stack.pop()
        st_idx = ST2IDX[st]

        #print("Traceback:", st, st_idx, i,j) # tracebackをチェックしたい時は、ここをコメントアウトする。 st,S/T/U st_idx:0,1,2
        next = TR[st_idx][i][j]

        if type(next) is list: # リスト型であれば、bifucation
            # 塩基は出力しない
            (st2_idx, i2, j2), (st3_idx, i3, j3) = next
            #print("bifucation:", "(", IDX2ST[st2_idx], i2, j2,")", "(",IDX2ST[st3_idx], i3, j3,")")
            stack.append((IDX2ST[st2_idx], i2, j2))
            stack.append((IDX2ST[st3_idx], i3, j3))
        elif(type(next) is tuple):
            if np.isnan(next[0]):
                print(f"Traceback error: you accessed undefined element at {i,j} in TR matrix", file=sys.stderr) 
                exit(0)

            # 塩基の出力
            base = TB[st_idx][i][j]
            if st == 'S':
                if base != '':
                    seq[i] = base
                    ss[i]  = '.'
            elif st == 'T':
                if base != '':
                    seq[j] = base
                    ss[j]  = '.'
            elif st == 'U':
                seq[i],seq[j] = base[0],base[1]
                ss[i],ss[j]  = '(',')'

            if i == j:
                continue

            # 次のステートへの遷移
            (st2_idx, i2, j2) = next
            stack.append((IDX2ST[st2_idx], i2, j2))
        else:
            print(f"Unknown type of class obtained by traceback {next}", file=sys.stderr)
            exit(0)

    #        st, i, j = IDX2ST[st2_idx], i2, j2

    gen_seq = ''.join(seq)
    gen_ss  = ''.join(ss)
    #print(gen_seq)
    #print(gen_ss)
    print(gen_seq, file=sys.stderr)
    print(gen_ss, file=sys.stderr)


def outputScoreMatrix(score, png_name):
    #s = score.shape
    #if s[0] != s[1]:
    #    print(f"Input matrix is not square ({s[0]}x{s[1]})", file=sys.stderr)
    #    exit(0)
                    
    sns.heatmap(score)
    plt.savefig(png_name)
    plt.close()

def readInput(fname: str):
    d_seq = {}
    d_ss  = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            sid, seq, ss = line.split(' ')
            d_seq[sid] = seq
            d_ss[sid]  = ss
    return d_seq, d_ss

class Dataset:
    def __init__(self, input, sid_list): # input_data is list of tensor
        self.data   = input
        self.sid_list = sid_list

    def __getitem__(self, index):
        return self.data[index], self.sid_list[index]
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('n', help='number of RNA to be produced')
    parser.add_argument('model', help='trained VAE model')
    parser.add_argument('outfile', help='output file namel')
    parser.add_argument('--input', help='input data file')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')    
    parser.add_argument('--matout', help='matrix output directory')
    parser.add_argument('--outProb', action='store_true', help='output TtoU probability matrix')
    parser.add_argument('--out_fasta', action='store_true', help='output fasta file')
    #parser.add_argument('--nuc_only', action='store_true', help='nucleotide only training')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of CPU to use')
    parser.add_argument('--from_emb', action='store_true', help='input embedding directly')
    args = parser.parse_args()

    main(args)

    
