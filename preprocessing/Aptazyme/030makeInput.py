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
    sid2act = {}
    with open(args.act) as f:
        for line in f:
            line = line.replace('\n','')
            #sid, act1, act2 = line.split(' ')
            #act = float(act1)
            sid, act = line.split(' ')
            act = float(act)
            sid2act[sid] = act

    with open(args.ss) as f:
        for line in f:
            line = line.replace('\n','')
            if line[0] == '>':
                sid = line[1:]
                seq = next(f).replace('\n','')
                items  = next(f).replace('\n','').split(' ')
                ss = items[0]
                if sid in sid2act:
                    print(sid, seq, ss)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ss', help='sequence and ss predicted by centroidfold')
    parser.add_argument('act', help='activity information')
    args = parser.parse_args()

    main(args)
    
