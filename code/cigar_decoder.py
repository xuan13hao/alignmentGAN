# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:14:08 2023
"""
import sys
import torch
from itertools import product
MAXINT = 10000

def decode_edit(batch_size=1):
    model = torch.load("pretrained_gen_cigar.pkl")#pretrained_gen_edits edit_generator
    out = model.sample(batch_size)
    # dict = decode_edits_dic()
    reads_file = open("../benchmark/cigar.fa", 'w')
    # print(out)
    seqs = []
    idx = 0
    for i in out:
        print(i)
        # idx = idx + 1
        # l = []
        # for n in i:
        #     # print(n)
        #     l.append(str(n))
        # print(l)
        # seq = ""
        # for i in l:
        #     seq = seq + i
        # # seq = generate_sequence(l)
        # # seqs.append(">"+idx+"\n"+seq+"\n")
        # reads_file.write(">"+str(idx)+"\n"+seq+"\n")
        # print(">",idx)
        # print(seq)
        # seqs.append(l)
    # print("gen edits num = ", len(seqs))
        


if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    decode_edit(batch_size)
    # print(result[0])
    