import os
import sys
import torch

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

def get_token2idx(nuc_letters): # Letterと行番号の辞書を作成
    d = {}
    for i,x in enumerate(nuc_letters):
        d[x] = i
    return d

def readAct(fname):
    sid2act = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            items = line.split() # 可変にしておく
            sid, act = items[0], items[1]
            act = float(act)
            #print(type(sid))
            #exit(0)
            sid2act[sid] = act
    return sid2act


class Dataset:
    def __init__(self, input_mat, sid_list, act_list): # input_data is list of tensor 
        self.data   = input_mat
        self.sid_list = sid_list
        self.act_list = torch.tensor(act_list, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data[index], self.sid_list[index], self.act_list[index]
    
    def __len__(self):
        return len(self.data)

