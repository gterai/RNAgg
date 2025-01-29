import sys
#import os
import SS2shape3
import numpy as np
import torch
import torch.nn.functional as F

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
            g_mat[i,0] = 1. 
            g_mat[j,1] = 1. 
        elif r_type == 2: # S->T
            i, j = r[1]
            g_mat[i,2] = 1.
            g_mat[j,3] = 1.
        elif r_type == 3: # T->Ta
            i, j = r[1]
            g_mat[i,4] = 1. 
            g_mat[j,5] = 1. 
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

    return g_mat

def makeMatrix(sid2seq, sid2ss, sid_list, word_size, g_dim, token2idx, nuc_yes_no):

    max_len = max([len(x) for x in sid2seq.values()])
    
    B_mat = [] # binary_matrix
    
    #word_size = len(NUC_LETTERS) 
    #VDIM = word_size + G_DIM # バイナリベクトルの次元
    
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
        g_mat = torch.tensor(getGramMatrix(r, max_len, g_dim)).float() # 文法のdimensionはG_DIM
        
        # 塩基の行列を作成する。
        nuc_token_list = []
        for i in range(max_len):
            token_id = token2idx[seq[i]]
            nuc_token_list.append(token_id)
        n_mat = F.one_hot(torch.tensor(nuc_token_list), word_size).float()
        #print(n_mat)
        
        if nuc_yes_no == 'yes':
            B_mat.append(n_mat)
        elif nuc_yes_no == 'no':
            tmp_mat = torch.cat((n_mat, g_mat), dim=1)
            B_mat.append(tmp_mat)
        else:
            print(f"Unexpected value of nuc_yes_no: {nuc_yes_no}", file=sys.stderr)
            exit(0)
            
            
        # 入力行列の確認
        if(0):
            sns.heatmap(input_mat)
            plt.savefig("input.png")
            plt.close()
            exit(0)
    
    return B_mat
