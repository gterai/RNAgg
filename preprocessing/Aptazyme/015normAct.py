# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
#sys.path.append("/home/terai/pyscript")
#import basic
#from Bio import SeqIO
#from Bio.SeqRecord import SeqRecord
#import re
#import csv
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import svm
#from scipy.stats import pearsonr
#import RNA

def main(args: dict):
    
    # xlsxファイルからデータを読み込む
    df = pd.read_table(args.raw, sep=' ', index_col=0, header=None)
    #print(df)

    max_val = df[1].max()
    min_val = df[1].min()

    #print(max_val, min_val)

    for mut in df.index:
        val = df.loc[mut][1]
        norm_val = (val - min_val)/(max_val - min_val)
        print(mut, norm_val)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('raw', help='the act_apt_raw.txt file')
    args = parser.parse_args()

    main(args)
    

