# -*- coding: utf-8 -*-
"""
Kmer
"""
from itertools import chain
import pandas as pd
SEQ_LENGTH = 101
import fasta
# ,mask_token = 'MASK'
# L - K + 1    200 - 3 + 1
def DNAToWord(dna, K,pad_token='PAD',mask_token = 'MASK'):
    kmers = []
    length = len(dna)
    for i in range(length - K + 1):
        # sentence += dna[i: i + K] + " "
        seq = dna[i: i + K].upper()
        # if 'N' in seq:
        #     seq = mask_token
        kmers.append(seq) 
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
    seqs.to_pickle(outputFilename)
    # list1 = []
    # for DNA in data:
    #     DNA = str(DNA).upper()
    #     list1.append(DNAToWord(DNA,kmer).split(" "))
    return seqs


#%%
if __name__ == '__main__':

    kmer_list = load_seqs('real.fa',1,outputFilename = "kmer.pkl")
    ref_list = load_seqs('ref.fa',1,outputFilename = "reference.pkl")
    # test = load_seqs('../data/test.fa',1,outputFilename = "test.pkl")
    # print(test)
