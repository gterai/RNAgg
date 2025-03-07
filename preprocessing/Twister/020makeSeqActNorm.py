# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
import pandas as pd

def main(args: dict):
    
    df = pd.read_table(args.act_raw, sep=" ", header=None)
    df.columns = ['seq', 'act_raw'] 

    #print(df)
    max_raw, min_raw = df['act_raw'].max(), df['act_raw'].min()

    pref = "GGGCCGCCU"

    # open files
    fseq = open(args.out_seq, "w")
    fact = open(args.out_act, "w")
    
    for i in df.index:
        seq = pref + df.loc[i]['seq']
        norm_act = (df.loc[i]['act_raw'] - min_raw)/(max_raw - min_raw)

        print(f">{i}", file=fseq)
        print(seq, file=fseq)
        
        print(i, norm_act, file=fact)
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('act_raw', help='the twister_act_raw.txt file')
    parser.add_argument('out_seq', help='sequence fasta file')
    parser.add_argument('out_act', help='normalized activity file')
    args = parser.parse_args()

    main(args)
    
