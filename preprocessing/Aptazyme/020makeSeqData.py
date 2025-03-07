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
    
    df = pd.read_table(args.act, index_col=0, header=None, sep=" ")
    #template = "TAATACGACTCACTATAGGGTCGNNNNNNNATAATCGCGTGGATATGGCACGCAAGTTTCTACCGGGCACCGTAAATGTCCGACTGGAGCCGTTCGGGCGGCTATAAACAGACCTCAGGCCCGAAGCGTGGCGGCACCTGCCGCCGGTGGTAAAAAAGATCGGAAGAGCACACGTCT".replace('T','U')
    template = "GGGTCGNNNNNNNATAATCGCGTGGATATGGCACGCAAGTTTCTACCGGGCACCGTAAATGTCCGACTGGAGCCGTTCGGGCGGCTATAAACAGACCTCAGGCCCGAAGCGTGGCGGCACCTGCCGCCGGTGGTAAAAAAGATCGGAAGAGCACACGTCT".replace('T','U')
    
    for mix in df.index:
        seq = template.replace('NNNNNNN', mix)
        print(f">{mix}")
        print(seq)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('act', help='the act_apta_norm.txt file')
    args = parser.parse_args()

    main(args)
    
