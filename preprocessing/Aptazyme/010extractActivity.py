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
    df = pd.read_excel(args.xlsx, skiprows=[0, 1])
    
    for i in list(df.index):
        print(df.iloc[i]['RNA seq'], df.iloc[i]['FC-'], df.iloc[i]['FC+'])

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='excel file from Pistol Ribozyme paper')
    args = parser.parse_args()

    main(args)
    

