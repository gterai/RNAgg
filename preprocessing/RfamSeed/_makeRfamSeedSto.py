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
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import svm
#from scipy.stats import pearsonr
#import RNA

def main(args: dict):
    
    with open(args.seed, encoding='iso-8859-1') as f:
        data = []
        for line in f:
            line = line.replace('\n','') 
            if line.startswith('#=GF AC'):
                fam_id = line.split()[2].strip()
            
            if line.startswith('//'): # fam_idの行を取得するため、elifとしない。
                data.append(line)
                if fam_id == args.fam_id:
                    for l in data:
                        print(l)
                data = []
            else:
                data.append(line)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('fam_id', help='family id')
    parser.add_argument('seed', help='Rfam.seed file')
    #parser.add_argument('n_seq', type=int, help='number of sequences to be output')
    #parser.add_argument('out_fasta', help='output fasta file name')
    args = parser.parse_args()

    main(args)
    
