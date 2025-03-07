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

    with open(args.ss) as f:
        for line in f:
            line = line.replace('\n','')
            if line[0] == '>':
                sid = line[1:]
                seq = next(f).replace('\n','')
                items  = next(f).replace('\n','').split(' ')
                ss = items[0]
                #if sid in sid2act:
                seq = seq.replace('T', 'U')
                print(sid, seq, ss)

                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ss', help='The ss_mxfold2.txt file')
    args = parser.parse_args()

    main(args)
    
