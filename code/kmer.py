# -*- coding: utf-8 -*-
"""
Kmer 
Created on Thu March 10 11:14:08 2023
"""
import pandas as pd
from Bio import SeqIO
from itertools import islice
from tqdm import tqdm
from itertools import chain
import pandas as pd
import pickle
SEQ_LENGTH = 96 
import fasta
# ,mask_token = 'MASK'
# L - K + 1    101 - 3 + 1 = 99 
def DNAToWord(dna, K ,pad_token='PAD',mask_token = 'MASK'):
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

# def load_seqs(inputFilename,kmer,outputFilename='kmer.pkl'):
#     data = []
#     for record in fasta.parse(inputFilename):
#         data.append(str(record.seq))
#     data = pd.DataFrame(data)
#     data.columns = ['data']
#     # print(DNAToWord(data['data'],kmer))
#     seqs = data.apply(lambda row: DNAToWord(row['data'],kmer),axis=1)
#     coln = ['token'+str(x) for x in seqs.columns.tolist()]
#     seqs.columns = coln
#     # print(seqs)
#     seqs.to_pickle(outputFilename)
#     # list1 = []
#     # for DNA in data:
#     #     DNA = str(DNA).upper()
#     #     list1.append(DNAToWord(DNA,kmer).split(" "))
#     return seqs

def load_seqs(inputFilename, kmer, outputFilename='kmer.pkl'):
    with open(outputFilename, 'wb') as output_file:
        for record in fasta.parse(inputFilename):
            data = str(record.seq)
            seqs = DNAToWord(data, kmer)
            pickle.dump(seqs, output_file)

def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = [int(s) for s in list(line.strip().split())]
        lis.append(l)
    return lis
def save_kmers(filename):
    with open(filename, 'rb') as input_file:
        kmers = []
        while True:
            try:
                kmer = pickle.load(input_file)
                kmers.append(kmer)
            except EOFError:
                break
    kmer_size = len(kmers)
    return kmer_size

#%%
if __name__ == '__main__':

    # kmer_list = load_seqs('real_1000.fa',3,outputFilename = "kmer.pkl")
    # ref_list = load_seqs('ref_1000.fa',3,outputFilename = "reference.pkl")
    kmer_list = load_seqs('mt_10.fa',6,outputFilename = "kmer.pkl")
    ref_list = load_seqs('mt_kmers.fa',6,outputFilename = "reference.pkl")
    # k = extract_kmers("kmer.pkl")
    # ref_list = load_seqs('chrY_kmers.fa',6,outputFilename = "reference.pkl")
    real_kmer = save_kmers("kmer.pkl")
    ref_kmer = save_kmers("reference.pkl")
    # kmer_list = load_seqs('real_reads.fa',3,outputFilename = "kmer.pkl")
    # ref_list = load_seqs('real_ref.fa',3,outputFilename = "reference.pkl")
    # test = load_seqs('../data/test.fa',1,outputFilename = "test.pkl")
    print("real_kmer num = ", real_kmer)
    print("ref_kmer num = ", ref_kmer)
    # print(len(kmer_list))
    # dic = generate_all_kmers(1)
    # print(dic)
