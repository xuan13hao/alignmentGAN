# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:14:08 2023
"""
import sys
import torch
from itertools import product
MAXINT = 10000

def decode_edits():
    kmer_dict = {}
    kmer_dict[1] = "M"
    kmer_dict[2] = "D"
    kmer_dict[3] = "I"
    kmer_dict[4] = "X"
    return kmer_dict
def decode_edit(batch_size=1):
    model = torch.load("pretrained_gen_edits.pkl")#pretrained_gen_edits edit_generator
    out = model.sample(batch_size)
    dict = decode_edits()
    reads_file = open("../benchmark/cigar.fa", 'w')
    # print(out)
    seqs = []
    idx = 0
    for i in out:
        l = []
        for n in i:
            l.append(dict[int(n)])
        print(l)
        seq = ""
        for i in l:
            seq = seq + i
        idx = idx + 1
        seqs.append(">"+str(idx)+"\n"+seq+"\n")
        reads_file.write(">"+str(idx)+"\n"+seq+"\n")
        


if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    decode_edit(batch_size)
    # print(result[0])
    