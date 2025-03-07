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
    
    with open(args.data) as f:
        for line in f:
            line = line.replace('\n','')
            if line[0] == "#" or line[0:3] == "Num":
                continue
            else:
                num, pos, nuc, seq, fit, perc2 = line.split('\t')
                sid = '-'.join([num,pos,nuc])
                sid = sid.replace(' ', '+')
                print(f">{sid}")
                print(seq)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='The FitnessData.txt file')
    args = parser.parse_args()

    main(args)
    
