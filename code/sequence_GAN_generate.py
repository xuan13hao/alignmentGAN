# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:14:08 2023
"""
import sys
import torch
from itertools import product

def decode_all_kmers(k):
    alphabet = "ACGT"
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    # print(kmers)
    idx = 0
    kmer_dict = {}
    for kmer in kmers:
        idx = idx + 1
        kmer_dict[idx] = kmer
    return kmer_dict

def generate_sequence(kmer_list):
<<<<<<< HEAD
    print(kmer_list)
=======
    # print(kmer_list)
>>>>>>> 87e3b5fcfa6bec1dfa7d1132ca0f91035da96161
    sequence = kmer_list[0]
    k = len(kmer_list[0])
    for i in range(1, len(kmer_list)):
        sequence += kmer_list[i][k-1:]
    return sequence

def decode(k,batch_size=1):
<<<<<<< HEAD
    model = torch.load('generator.pkl')
=======
    model = torch.load('generator_4mer.pkl')
>>>>>>> 87e3b5fcfa6bec1dfa7d1132ca0f91035da96161
    out = model.sample(batch_size)
    dict = decode_all_kmers(k)
    reads_file = open("../benchmark/gen.fa", 'w')
    # print(out)
    seqs = []
    idx = 0
    for i in out:
        idx = idx + 1
        l = []
        for n in i:
            l.append(dict[int(n)])
        seq = generate_sequence(l)
        # seqs.append(">"+idx+"\n"+seq+"\n")
        reads_file.write(">"+str(idx)+"\n"+seq+"\n")
        # print(">",idx)
        # print(seq)
    

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    
<<<<<<< HEAD
    decode(6, batch_size)
=======
    decode(4, batch_size)
>>>>>>> 87e3b5fcfa6bec1dfa7d1132ca0f91035da96161
    # print(result[0])
    