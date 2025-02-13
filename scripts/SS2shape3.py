# -*- coding: utf-8 -*-

import os
import sys
import argparse
#import graphviz

def main(args: dict):
    sid2seq, sid2ss  = readInput(args.input)
    sid2rules = {}
    for sid in sid2seq.keys():
        seq, ss = sid2seq[sid].lower(), sid2ss[sid]
        bp_pos = getBPpos_ij(ss)
        rules = []
        (i,j) = (0,len(sid2seq[sid])-1)
        generate_rule_G4b(0, rules, seq, ss, bp_pos, (i,j), 'S')
        print(rules)
        #def generate_rule_G4b(ID, r, seq, ss, bp, pos, st): # STはステートのこと
        #gr = makeGraph(rules) # これは後回しにしよう。グラフで表示できると楽しいかも。
        #gr.render(args.outdir + '/' + str(sid))
        
        #print(sid, rules)
        #sid2rules[sid] = rules

        #gr = makeGraph(rules)
        #outputGraph(gr, f'{sid}.png')
        #gr.render('test')
        #exit(0)


def getBPpos_ij(ss: str):
    bp = ['' for i in range(len(ss))]
    stack = []
    for j in range(len(ss)):
        if ss[j] == '.':
            bp[j] = -1
        elif ss[j] == '(':
            stack.append(j)
        elif ss[j] == ')':
            if len(stack) == 0:
                print("base pair parse error", file=sys.stderr)
                exit(0)
            i = stack.pop()
            #print("ok",i,j)
            bp[i] = j
            bp[j] = i # ここが違うだけと思われる。
        else:
            print("parsing error", file=sys.stderr)
            exit(0)
    #print(bp)
    if len(stack) > 0:
        print("base pair parse error", file=sys.stderr)
        exit(0)
    return bp


#single_stype = {'A':1, 'C':2, 'G':3, 'U':4, 'X':5}
#pair_stype = {'AU':1, 'UA':2, 'GC':3, 'CG':4, 'GU':5, 'UG':6}
#single_stype = {'a':1, 'c':2, 'g':3, 'u':4, '-':5, 'x':6}     # このあたりの数字は使うのだろうか？
#pair_stype = {'au':1, 'ua':2, 'gc':3, 'cg':4, 'gu':5, 'ug':6} # このあたりの数字は使うのだろうか？

class SCFGParseError(Exception):
    pass

def generate_rule_G4b(ID, r, seq, ss, bp, pos, state): # STはステートのこと
    # 引数の説明
    # ID:ルールの連番, r:ルールのリスト, seq:RNA配列, ss:2時構造
    # bp:塩基対位置,  pos:(i,j), state:非終端記号

    # i=jの時のルールも統一している。
    # rule_type
    # 1(ss): S->aS
    # 2(st): S->T
    # 3(tt): T->Ta
    # 4(tu): T->U
    # 5(tb): T->TU
    # 6(us): U->aSb

    # rule_subtype
    # A|C|G|U = 1|2|3|4
    # AU|UA|GC|CG|GU|UG = 1|2|3|4|5|6
    
    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。
    i,j = pos

    # rに含まれる情報が冗長。
    # 出力される（塩基|塩基対|分岐点）、コメントだけで良いのではないか。
    if state == 'S':
        if ss[i] == '.':     # 左側が'.'    S->aS
            r[ID] = [1, (i,j), seq[i], f'S->{seq[i]}S'] 
            if i != j:
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i+1,j), 'S') 
        else:     # 左側が'(|.|)'     S->T
            r[ID] = [2, (i,j), '', 'S->T'] 
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j), 'T')
    elif state == 'T':
        if ss[i] == '.':     # 左側が'.'
            print(f'State T must not be used when ss[i] is "."', file=sys.stderr)
            raise(SCFGParseError("Parsing failure"))
            #exit(0)
        elif ss[j] == '.':     # 右側が'.'    T->Ta
            r[ID] = [3, (i,j), seq[j], f'T->T{seq[j]}']
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j-1), 'T') 
        elif ss[j] == ')':     # 右側が')'
            if bp[j] == i:  # (any)        T->U
                r[ID] = [4, (i,j), '', 'T->U']
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j), 'U')
            else:           # any(any)     T->TU
                r[ID] = [5, (i,j), bp[j]-1, 'T->TU'] # bp[j]はbranching point + 1
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i,bp[j]-1), 'T')
                ID2 = len(r)
                generate_rule_G4b(ID2, r, seq, ss, bp, (bp[j],j), 'U')
        else:
            print(f'Unknown error at {i,j}, state={state}, ss[i] and ss[j]={ss[i], ss[j]}', file=sys.stderr)
            print(seq, file=sys.stderr)
            print(ss, file=sys.stderr)
            raise(SCFGParseError("Parsing failure"))
            #exit(0)
    elif state == 'U':
        if bp[j] == i:  # (any)        U->aUb
            r[ID] = [6, (i,j), seq[i]+seq[j],  f'U->{seq[i]}S{seq[j]}']
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i+1,j-1), 'S')
    else:
        print(f'Unexpected state {state}', file=sys.stderr)
        raise(SCFGParseError("Parsing failure"))
        #exit(0)


##################################################################
########################## 以下、不要 ############################
##################################################################

def generate_rule_G4b_old(ID, r, seq, ss, bp, pos, state): # STはステートのこと
    # 引数の説明
    # ID:ルールの連番, r:ルールのリスト, seq:RNA配列, ss:2時構造
    # bp:塩基対位置,  pos:(i,j), state:非終端記号

    # i=jの時のルールも統一している。
    # rule_type
    # 1(ss): S->aS
    # 2(st): S->T
    # 3(tt): T->Ta
    # 4(tu): T->U
    # 5(tb): T->TU
    # 6(us): U->aSb

    # rule_subtype
    # A|C|G|U = 1|2|3|4
    # AU|UA|GC|CG|GU|UG = 1|2|3|4|5|6
    
    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。
    i,j = pos

    # rに含まれる情報が冗長。
    # 出力される（塩基|塩基対|分岐点）、コメントだけで良いのではないか。
    if state == 'S':
        if ss[i] == '.':     # 左側が'.'    S->aS
            r[ID] = [1, (i,j), 'S', f'{seq[i]}S', i, single_stype[seq[i]]] # これは冗長
            if i != j:
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i+1,j), 'S') 
        else:     # 左側が'(|.|)'     S->T
            r[ID] = [2, (i,j), 'S', 'T', '', ''] # サブタイプなし
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j), 'T')
    elif state == 'T':
        if ss[i] == '.':     # 左側が'.'
            print(f'State T must not be used when ss[i] is "."', file=sys.stderr)
            raise(SCFGParseError("Parsing failure"))
            #exit(0)
        elif ss[j] == '.':     # 右側が'.'    T->Ta
            r[ID] = [3, (i,j), 'T', f'T{seq[j]}', j, single_stype[seq[j]]]
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j-1), 'T') 
        elif ss[j] == ')':     # 右側が')'
            if bp[j] == i:  # (any)        T->U
                r[ID] = [4, (i,j), 'T', f'U', '', '']
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j), 'U')
            else:           # any(any)     T->TU
                r[ID] = [5, (i,j), 'T', 'TU', '', bp[j]-1] # bp[j]はbranching point + 1
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i,bp[j]-1), 'T')
                ID2 = len(r)
                generate_rule_G4b(ID2, r, seq, ss, bp, (bp[j],j), 'U')
        else:
            print(f'Unknown error at {i,j}, state={state}, ss[i] and ss[j]={ss[i], ss[j]}', file=sys.stderr)
            print(seq, file=sys.stderr)
            print(ss, file=sys.stderr)
            raise(SCFGParseError("Parsing failure"))
            #exit(0)
    elif state == 'U':
        if bp[j] == i:  # (any)        U->aUb
            r[ID] = [6, (i,j), 'U', f'{seq[i]}S{seq[j]}', (i,j), pair_stype[seq[i]+seq[j]]]
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i+1,j-1), 'S')
    else:
        print(f'Unexpected state {state}', file=sys.stderr)
        raise(SCFGParseError("Parsing failure"))
        #exit(0)


def generate_rule_G4b_org(ID, r, seq, ss, bp, pos, st): # STはステートのこと
    # i=jの時のルールも統一している。
    # rule_type
    # 1: S->aS
    # 2: S->T
    # 3: T->Ta
    # 4: T->U
    # 5: T->TU
    # 6: U->aSb

    # rule_subtype
    # A|C|G|U = 1|2|3|4
    # AU|UA|GC|CG|GU|UG = 1|2|3|4|5|6

    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。
    i,j = pos

    if st == 'S':
        if ss[i] == '.':     # 左側が'.'    S->aS
            r[ID] = [1, (i,j), 'S', f'{seq[i]}S', i, single_stype[seq[i]]]
            if i != j:
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i+1,j), 'S') 
        else:     # 左側が'(|.|)'     S->T
            r[ID] = [2, (i,j), 'S', 'T', '', ''] # サブタイプなし
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j), 'T')
    elif st == 'T':
        if ss[i] == '.':     # 左側が'.'
            print(f'State T must not be used when ss[i] is "."', file=sys.stderr)
            exit(0)
        elif ss[j] == '.':     # 右側が'.'    T->Ta
            r[ID] = [3, (i,j), 'T', f'T{seq[j]}', j, single_stype[seq[j]]]
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j-1), 'T') 
        elif ss[j] == ')':     # 右側が')'
            if bp[j] == i:  # (any)        T->U
                r[ID] = [4, (i,j), 'T', f'U', '', '']
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i,j), 'U')
            else:           # any(any)     T->TU
                r[ID] = [5, (i,j), 'T', 'TU', '', bp[j]-1] # bp[j]はbranching point + 1
                generate_rule_G4b(ID+1, r, seq, ss, bp, (i,bp[j]-1), 'T')
                ID2 = len(r)
                generate_rule_G4b(ID2, r, seq, ss, bp, (bp[j],j), 'U')
        else:
            print(f'Unknown error at {i,j}, state={st}, ss[i] and ss[j]={ss[i], ss[j]}', file=sys.stderr)
            print(seq, file=sys.stderr)
            print(ss, file=sys.stderr)
            exit(0)
    elif st == 'U':
        if bp[j] == i:  # (any)        U->aUb
            r[ID] = [6, (i,j), 'U', f'{seq[i]}S{seq[j]}', (i,j), pair_stype[seq[i]+seq[j]]]
            generate_rule_G4b(ID+1, r, seq, ss, bp, (i+1,j-1), 'S')
    else:
        print(f'Unexpected state {st}', file=sys.stderr)
        exit(0)


"""
def generate_rule_G4a(ID, r, seq, ss, bp, pos, st):
    # rule_type
    # 1: S->aS
    # 2: S->T
    # 3: T->Ta
    # 4: T->aSb
    # 5: T->TU
    # 6: U->aSb
    # 7: S->a # i == jのときだけ使うルール

    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。
    i,j = pos

    # i == jのとき
    if i == j: 
        if st != 'S':
            print(f'Unexpected State {st} at position ({i,j})', file=sys.stderr)
            exit(0)

        if ss[i] == '.':     # ss[i]が'.'でないときはエラー
            r[ID] = [7, (i,j), 'S1', f'{seq[i]}', i, single_stype[seq[i]]] # S->a
        else:
            print(f'Position ({i,j}) must be single stranded, but was ({ss[i]})', file=sys.stderr)
            exit(0)

        return # important !
    
    # i != jのとき
    if st == 'S':
        if ss[i] == '.':     # 左側が'.'    S->aS
            r[ID] = [1, (i,j), 'S', f'{seq[i]}S', i, single_stype[seq[i]]]
            generate_rule_G4a(ID+1, r, seq, ss, bp, (i+1,j), 'S') 
        else:     # 左側が'(|.|)'     S->T
            r[ID] = [2, (i,j), 'S', 'T', '', ''] # サブタイプなし
            generate_rule_G4a(ID+1, r, seq, ss, bp, (i,j), 'T')
    elif st == 'T':
        if ss[i] == '.':     # 左側が'.'
            print(f'State T must not be used when ss[i] is "."', file=sys.stderr)
            exit(0)
        elif ss[j] == '.':     # 右側が'.'    T->Ta
            r[ID] = [3, (i,j), 'T', f'T{seq[j]}', j, single_stype[seq[j]]]
            generate_rule_G4a(ID+1, r, seq, ss, bp, (i,j-1), 'T') 
        elif ss[j] == ')':     # 右側が')'
            if bp[j] == i:  # (any)        T->aSa
                r[ID] = [4, (i,j), 'T', f'{seq[i]}S{seq[j]}', (i,j), pair_stype[seq[i]+seq[j]]]
                generate_rule_G4a(ID+1, r, seq, ss, bp, (i+1,j-1), 'S')
            else:           # any(any)     T->TU
                r[ID] = [5, (i,j), 'T', 'TU', '', ''] # サブタイプなし
                generate_rule_G4a(ID+1, r, seq, ss, bp, (i,bp[j]-1), 'T')
                ID2 = len(r)
                generate_rule_G4a(ID2, r, seq, ss, bp, (bp[j],j), 'U')
        else:
            print(f'Unknown error at {i,j}, state={st}, ss[i] and ss[j]={ss[i], ss[j]}', file=sys.stderr)
            exit(0)
    elif st == 'U':
        if bp[j] == i:  # (any)        U->aUb
            r[ID] = [6, (i,j), 'U', f'{seq[i]}S{seq[j]}', (i,j), pair_stype[seq[i]+seq[j]]]
            generate_rule_G4a(ID+1, r, seq, ss, bp, (i+1,j-1), 'S')
    else:
        print(f'Unexpected state {st}', file=sys.stderr)
        exit(0)
"""


"""
def generate_rule_G4(ID, r, seq, bp, pos, nt):
    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。
    #print(nt, pos)
    i,j = pos
    if nt == 'S':
        if bp[i] == -1:     # 左側が'.'    S->Sa
            generate_rule_G4(ID+1, r, seq, bp, (i+1,j), 'S') 
            r[ID] = [f'{seq[i]}S', i]
        elif bp[i] > i:     # 左側が'('    S->T
            generate_rule_G4(ID+1, r, seq, bp, (i,j), 'T')
            r[ID] = ['T']
        elif bp[i] < i:     # 左側が')'    S->e
            r[ID] = ['e']
        else:
            print(f'Unknown error {i,j}', file=sys.stderr)
            exit(0)
    elif nt == 'T':
        #print(j)
        if bp[j] == -1:     # 右側が'.'    T->Ta
            generate_rule_G4(ID+1, r, seq, bp, (i,j-1), 'T') 
            r[ID] = [f'T{seq[j]}', j]
        elif bp[j] < j:     # 右側が')'
            if bp[j] == i:  # (any)        S->aSa
                generate_rule_G4(ID+1, r, seq, bp, (i+1,j-1), 'S')
                r[ID] = [f'{seq[i]}S{seq[j]}', f'{i}:{j}']
            else:           # any(any)     S->TaSa
                generate_rule_G4(ID+1, r, seq, bp, (i,bp[j]-1), 'T')
                ID2 = len(r)
                generate_rule_G4(ID2, r, seq, bp, (bp[j]+1,j-1), 'S')
                r[ID] = [f'T{seq[bp[j]]}S{seq[j]}', f'{bp[j]}:{j}', ID+1, ID2]
        elif bp[j] > j:     # 右側が'('    
            print(f'In T->*, \'(\' must not be found', file=sys.stderr)
            exit(0)            
        else:
            print(f'Unknown error 2{i,j}', file=sys.stderr)
            exit(0)
    else:
        print(f'Unexpected non-terminal symbol {nt}', file=sys.stderr)
        exit(0)
"""

"""
def generate_rule_ss(ID, r, seq, bp, pos):
    # ssのみを考慮した文法でルールを作成
    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。
    i,j = pos
    #print(i,j)
    if j + 1 == i: # ssの左側が'.'
        r[ID] = ['e', 'e']
    elif bp[i] == -1: # ssの左側が'.'
        r[ID] = ['l', seq[i], i]
        generate_rule_ss(ID+1, r, seq, bp, (i+1,j)) # 左側出力
    elif bp[j] == -1: # ssの右側が'.'
        r[ID] = ['r', seq[j], j]
        generate_rule_ss(ID+1, r, seq, bp, (i,j-1)) # 右側出力
    elif bp[i] > 0: # iが塩基対を組む
        if bp[i] == j: # pair
            ID2 = len(r)
            pair = seq[i]+seq[j]
            r[ID] = ['p', seq[i], seq[j], (i, j)] 
            generate_rule_ss(ID2, r, seq, bp, (i+1,j-1))
        else:          # bifucation(塩基で囲まれた部分が左側)
            ID2 = len(r)
            generate_rule_ss(ID2, r, seq, bp, (i,bp[i]))
            ID3 = len(r)
            generate_rule_ss(ID3, r, seq, bp, (bp[i]+1,j))
            r[ID] = ['b', ID2, ID3]
    elif bp[i] == -2: # ssの左側が')'
            pass
    else:
        print(f'unexpected base pair information {bp[i]}', file=sys.stderr)
        exit(0)

"""

def readInput(fname: str):
    d_seq = {}
    d_ss  = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            sid, seq, ss = line.split(' ')
            d_seq[sid] = seq
            d_ss[sid]  = ss
            #print(sid, seq, ss)
    #exit(0)
    return d_seq, d_ss

def makeGraph(rules: list): # これは古いルールのフォーマットに基づいており、使うためには改良が必要(2024/11/20)
    # rulesにはTreeShapeの情報がリストとして記述されている。
    # この関数では、rulesを木構造にして、graphvisオブジェクトに格納する。
    # rulesのindexがノードのIDに対応する。
    # たとえば、rules[0]はrootノードである。
    
    gr = graphviz.Digraph(format='png')
    root_node_id = 0
    stack = [root_node_id]  # root nodeをスタックに入れる
    gr.node(nName(root_node_id))  # root nodeの作成

    #print(rules)
    #exit(0)

    while(len(stack) > 0):
        node_id = stack.pop()
        node_type = rules[node_id][0] # rulesの最初の要素にはルールのタイプがある(e, p, b)のいずれか
        if node_type == 'e': #ノードは塩基を出力する
            # 出力される塩基を取り出す
            nuc = rules[node_id][1]

            # 出力される塩基を１つのノードとして記録する
            nuc_node_id = f"{nuc} ({node_id})"
            gr.node(nuc_node_id)
            gr.edge(nName(node_id), nuc_node_id) # ノードとエッジを張る
            
        elif node_type == 'p': #ノードが[塩基対]と[次のノード]を出力する。
            # 出力される塩基対を取り出す
            (pL, pR) = rules[node_id][1:3] 
            
            # 出力される塩基対を１つのノードとして記録する
            pair_node_id = f"{pL}-{pR} ({node_id})"
            gr.node(pair_node_id)
            gr.edge(nName(node_id), pair_node_id) # ノードとエッジを張る
            
            # 次のノードを作成して、エッジを張る
            next_node_id = node_id + 1
            gr.node(nName(next_node_id))
            gr.edge(nName(node_id), nName(next_node_id))
            
            # 次のノードをスタックに入れて、次のループへ
            stack.append(next_node_id)
            
        elif node_type == 'b': #ノードが２つのノードに別れる
            L_node_id, R_node_id = rules[node_id][1:3] 

            # 左側ノードを作成して、エッジを張る
            gr.node(nName(L_node_id))
            gr.edge(nName(node_id), nName(L_node_id))

            # 右側ノードを作成して、エッジを張る
            gr.node(nName(R_node_id))
            gr.edge(nName(node_id), nName(R_node_id))

            # 左右のノードをスタックに入れて、次のループへ
            stack.append(L_node_id)
            stack.append(R_node_id)
            
        else:
            print("Error: unknown node type ({node_type})", file=sys.stderr)
            exit(0)
            
    return gr

            
def nName(i: int):
    return 'n' + str(i)


"""
def getBPpos(ss: str):
    bp = ['' for i in range(len(ss))]
    stack = []
    for j in range(len(ss)):
        if ss[j] == '.':
            bp[j] = -1
        elif ss[j] == '(':
            stack.append(j)
        elif ss[j] == ')':
            if len(stack) == 0:
                print("base pair parse error", file=sys.stderr)
                exit(0)
            i = stack.pop()
            #print("ok",i,j)
            bp[i] = j
            bp[j] = -2
        else:
            print("parsing error", file=sys.stderr)
            exit(0)
    #print(bp)
    if len(stack) > 0:
        print("base pair parse error", file=sys.stderr)
        exit(0)
    return bp
"""

"""
def generate_rule(ID, r, seq, bp, pos):
    r.append([]) # IDに対応するルールの箱だけ先に確保してしまう。

    i,j = pos
    #print(i,j)
    if bp[i] == -1: # ssの左側が'.'
        if i == j: # １塩基の出力
            r[ID] = ['e', seq[i], i]
        elif i < j: # bifucation
            ID2 = len(r)
            #r.append([]) # IDを生成するたびに、１つ箱を用意する。
            generate_rule(ID2, r, seq, bp, (i,i)) # そして、再起を開始する。
            ID3 = len(r)
            #r.append([]) # IDを生成するたびに、１つ箱を用意する。
            generate_rule(ID3, r, seq, bp, (i+1,j))
            r[ID] = ['b', ID2, ID3]
        else:
            print(f'position error {i,j}', file=sys.stderr)
            exit(0)
    elif bp[i] > 0:
        if bp[i] == j: # pair
            ID2 = len(r)
            #r.append([]) # IDを生成するたびに、１つ箱を用意する。
            pair = seq[i]+seq[j]
            if not pair in ['AU','UA','GC','CG','GU','UG']:
                print(f"Invalid pase pair {pair}", file=sys.stderr)
                exit(0) 
            r[ID] = ['p', seq[i], seq[j], (i, j)] # ここで塩基対のチェックを入れるべき
            #print("pair", i,j, "ID=", ID, "ID2=", ID2)
            if i+1 == j:
                print(f'At least one base is necessacy between a base pair {i,j}', file=sys.stderr)
                exit(0)
            generate_rule(ID2, r, seq, bp, (i+1,j-1))
        else:          # bifucation(塩基で囲まれた部分が左側)
            ID2 = len(r)
            #r.append([]) # IDを生成するたびに、１つ箱を用意する。
            generate_rule(ID2, r, seq, bp, (i,bp[i]))
            ID3 = len(r)
            #r.append([]) # IDを生成するたびに、１つ箱を用意する。
            generate_rule(ID3, r, seq, bp, (bp[i]+1,j))
            r[ID] = ['b', ID2, ID3]
    elif bp[i] == -2: # ssの左側が')'
            pass
    else:
        print(f'unexpected base pair information {bp[i]}', file=sys.stderr)
        exit(0)
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input sequence and ss file')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()

    main(args)
    
