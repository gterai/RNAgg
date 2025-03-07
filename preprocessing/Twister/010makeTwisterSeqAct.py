# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
import pandas as pd

def getWT(df):
    
    old = ''
    wt_seq = []
    for mut in df.index:
        new = mut[:-1]

        if old != new:
            wt_seq.append(new[0])
            old = new

    return wt_seq

def main(args: dict):
    
    df = pd.read_excel(args.xlsx, skiprows=[0, 1], index_col=0)
    
    wt_seq = getWT(df)
    #print(wt_seq)
    
    #print(len(df.index))
    #print(len(df.columns))
    #exit(0)
    for i, mut1 in enumerate(df.index):
        for j, mut2 in enumerate(df.index):
            if i < j:
                continue
            
            mut_seq = [n for n in wt_seq]
            
            if mut1 == mut2: # single point mutation
                wt, pos, mu = mut1[0], int(mut1[1:-1]), mut1[-1]
                pos -= 7

                if wt_seq[pos] != wt:
                    print(f"wt base is different at position {pos}. {wt_seq[pos]}:{wt}", file=sys.stderr)
                    exit(0)

                mut_seq[pos] = mu
                print(''.join(mut_seq), df[mut1][mut2]) 

            elif mut1[:-1] == mut2[:-1]: # 同じ場所の違うmutation
                pass
            else:
                wt1, pos1, mu1 = mut1[0], int(mut1[1:-1]), mut1[-1]
                wt2, pos2, mu2 = mut2[0], int(mut2[1:-1]), mut2[-1]
                pos1 -= 7
                pos2 -= 7
                
                if wt_seq[pos1] != wt1 or wt_seq[pos2] != wt2:
                    print(f"wt base is different at position {pos1,pos2}. {wt_seq[pos1]}:{wt1} {wt_seq[pos2]}:{wt2}", file=sys.stderr)
                    exit(0)

                mut_seq[pos1] = mu1
                mut_seq[pos2] = mu2
                print(''.join(mut_seq), df[mut1][mut2]) 
                
            
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='the Kobori_ACIE_2016_Supporting_Data.xlsx file')
    args = parser.parse_args()

    main(args)
    
