# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
#sys.path.append("/home/terai/pyscript")
#import basic
import re
import numpy as np
from Bio import AlignIO
#from SS2shape2 import generate_rule_G4b, SCFGParseError
from SS2shape3 import generate_rule_G4b, SCFGParseError, getBPpos_ij

def getBP(ss:list):
    bp_list = []
    stack = []
    for i in range(len(ss)):
        if ss[i] == '(' or ss[i] == '<' or ss[i] == '{' or ss[i] == '[':
            stack.append(i)
        elif ss[i] == ')' or ss[i] == '>' or ss[i] == '}' or ss[i] == ']':
            left = stack.pop()
            bp_list.append((left, i))
        elif ss[i] == '.' or ss[i] == ',' or ss[i] == ':' or \
             ss[i] == '_' or ss[i] == '-' or ss[i] == '~':
            pass
        elif ss[i] == 'A' or ss[i] == 'a' or \
             ss[i] == 'B' or ss[i] == 'b': # pseudo knot
            pass
        else:
            print(f"Unexpected SS annotation ({ss[i]})", file=sys.stderr)
            exit(0)
    
    return bp_list

valid_pair = ['AU','UA','GC','CG','GU','UG']
def main(args: dict):

    # 出力ファイルを先にオープンしておく
    f_una = open(args.outfile_una, mode='w')
    f_ali = open(args.outfile_ali, mode='w')
    
    align = AlignIO.read(args.stk, "stockholm")

    SS = align.column_annotations['secondary_structure']
    SS_list = [i for i in SS]
    #print(SS) # チェック

    # bp_listの取得
    bp_list = getBP(SS_list) # 共通2時構造の塩基対
    
    num_pairs = {}
    sid = 0
    max_len, min_len = -1e10, 1e10
    save_nr_set = set()
    for record in align:
        
        # AUGC-以外の塩基が含まれているかをチェック
        pattern = r'[^AUGC-]'
        seq_str = str(record.seq).upper()
        matches = re.findall(pattern, seq_str)
        if matches:
            print(f"{record.id} contains invalid letters {matches}.", file=sys.stderr) # cmalignの結果は小文字を含むので、取り除く。
            continue

        seq_list = [i for i in seq_str]

        # bp_listに含まれる塩基をチェック
        bp_list_pass = [] # AU, GC, GUで形成されるもの

        for bp in bp_list:
            pair = seq_list[bp[0]] + seq_list[bp[1]]
            if pair in valid_pair:
                bp_list_pass.append(bp)

                # pairを種類ごとに数えておく
                if pair not in num_pairs:
                    num_pairs[pair] = 0 # 動的初期化
                num_pairs[pair] += 1
        #print(bp_list)
        #print(bp_list_pass)

        # bp_list_passに従って、新しい２次構造アノテーションを作る
        SS_list_pass = ['.' for i in SS] # 配列長のドットで初期化
        for bp in bp_list_pass:
            SS_list_pass[bp[0]] = '('
            SS_list_pass[bp[1]] = ')'

        seq_nr, ss_nr = [], []
        for i in range(len(seq_list)):
            
            if seq_list[i] != '-':  # ギャップを取り除く(args.alignedがFalseのとき)
                seq_nr.append(seq_list[i])

                if SS_list_pass[i] == '(':
                    ss_nr.append('(')
                elif SS_list_pass[i] == ')':
                    ss_nr.append(')')
                elif SS_list_pass[i] == '.':
                    ss_nr.append('.')
                else:
                    print(f"Unexpected SS annotation ({SS_list_pass[i]})", file=sys.stderr)
                    exit(0)

        seq_str = ''.join(seq_nr)
        ss_str  = ''.join(ss_nr)

        ali_seq_str = ''.join(seq_list)
        ali_ss_str  = ''.join(SS_list_pass)
        
        # ギャップを除いた状態でG4 grammarでのparsingをチェックする。
        bp = getBPpos_ij(ss_str)
        rule = []
        try:
            generate_rule_G4b(0, rule, seq_str.lower(), ss_str, bp, (0,len(ss_str)-1), 'S') # ギャップを除いた状態でG4bでparseできるかをチェック
        except SCFGParseError as e:
            print(f"SCFGParse Error found in {record.id}.", file=sys.stderr)
            continue
        
        
        if max_len < len(seq_str):
            max_len = len(seq_str)
        if min_len > len(seq_str):
            min_len = len(seq_str)

        if seq_str in save_nr_set:
            print(f"Duplication of sequence {record.id}.", file=sys.stderr)
            continue
        else:
            save_nr_set.add(seq_str)
            
        print(f"{sid} {seq_str} {ss_str}", file=f_una)
        print(f"{sid} {ali_seq_str} {ali_ss_str}", file=f_ali)
        sid += 1
        
    #全pairの数を出力
    ordered = sorted(num_pairs.items(), key=lambda x:x[1], reverse=True)
    sum_pairs = np.sum(list(num_pairs.values()))

    for pair,num in ordered:
        print(pair, num/sum_pairs, file=sys.stderr)

    print(f"max len = {max_len}, min len = {min_len}, file = {args.stk}", file=sys.stderr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('stk', help='stockholm format file')
    parser.add_argument('outfile_una', help='unaligned input file name')
    parser.add_argument('outfile_ali', help='aligned input file name')
    #parser.add_argument('--aligned', action='store_true', help='make aligned sequence data')
    args = parser.parse_args()
    
    main(args)


