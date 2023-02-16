# -*- coding: utf-8 -*-
"""
Kmer
"""
from itertools import chain
import jieba
import pandas as pd
from config import PATH, SEQ_LENGTH
import fasta

# L - K + 1    200 - 3 + 1
def DNAToWord(dna, K,pad_token='PAD'):
    sentence = ""
    kmers = []
    length = len(dna)
    for i in range(length - K + 1):
        # sentence += dna[i: i + K] + " "
        kmers.append(dna[i: i + K].upper()) 
    if len(kmers) <= SEQ_LENGTH-1:
        kmers = kmers + [pad_token] * (SEQ_LENGTH - 1 - len(kmers))
    # sentence = sentence[0 : len(sentence) - 1]
    # kmers = [w for w in sentence] 
    # print(kmers)
    return pd.Series(kmers)

def load_seqs(inputFilename,kmer,outputFilename='kmer.pkl'):
    data = []
    for record in fasta.parse(inputFilename):
	    data.append(str(record.seq))
    data = pd.DataFrame(data)
    data.columns = ['data']
    # print(DNAToWord(data['data'],kmer))
    seqs = data.apply(lambda row: DNAToWord(row['data'],kmer),axis=1)
    coln = ['token'+str(x) for x in seqs.columns.tolist()]
    seqs.columns = coln
    # print(seqs)
    seqs.to_pickle(PATH+outputFilename)
    # list1 = []
    # for DNA in data:
    #     DNA = str(DNA).upper()
    #     list1.append(DNAToWord(DNA,kmer).split(" "))
    return seqs


#%%
if __name__ == '__main__':

    kmer_list = load_seqs('../data/test.fa',1)
    print(kmer_list)
