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
import numpy as np

def main(args: dict):
    
    sid_list = []
    fit_list = []
    with open(args.data) as f:
        for line in f:
            line = line.replace('\n','')
            if line[0] == "#" or line[0:3] == "Num":
                continue
            else:
                num, pos, nuc, seq, fit, perc2 = line.split('\t')
                sid = '-'.join([num,pos,nuc])
                sid = sid.replace(' ', '+')
                #print(f"{sid} {fit}")
                sid_list.append(sid)
                fit_list.append(float(fit))
                
    max_fit = np.max(fit_list)
    min_fit = np.min(fit_list)

    norm_fit_list = [ (x - min_fit)/(max_fit - min_fit) for x in fit_list]

    #max_fit = np.max(norm_fit_list)
    #min_fit = np.min(norm_fit_list)
    #print(max_fit, min_fit)

    for sid, norm_fit in zip(sid_list, norm_fit_list):
        print(sid, norm_fit)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='The FitnessData.txt file')
    args = parser.parse_args()

    main(args)
    
