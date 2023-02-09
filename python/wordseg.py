# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:21:03 2018
"""
from itertools import chain
import jieba
import pandas as pd
from config import PATH, SEQ_LENGTH
import fasta

def wordseg(x, pad_token='PAD'):
    try:
        text = list(jieba.cut(x))
        text = [w for w in text if w not in ['\n',' ']]
    except:
        text = []
    if len(text) <= SEQ_LENGTH-1:
        text = text + [pad_token] * (SEQ_LENGTH - 1 - len(text))
    
    # print(text)
    return pd.Series(text[0:SEQ_LENGTH-1])

def delSpace(x, ignored=[' ','\n','|','*']):
    for t in ignored:
        x = x.replace(t, '')
    return x
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
    # print(data)
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

def splitSentence(x, splitBy=['。', '！','；','……']):
    tmp = [x]
    for t in splitBy:
        tmp = [s.split(t) for s in tmp]
        if isinstance(tmp[0],list):
            tmp = list(chain.from_iterable(tmp))
    tmp = [x for x in tmp if len(x)>5]
    # print(tmp)
    return tmp

def readRandomText(inputFilename='london.txt',outputFilename='real_data_london.pkl'):
    lineList_all = list()
    with open(PATH+inputFilename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line.strip()
            lineList_all.append(line)
    # print(lineList_all)
    data = [delSpace(x) for x in lineList_all if len(x) > 5]
    # print(data)
    data = pd.DataFrame(list(chain.from_iterable([splitSentence(x) for x in data])))
    data.columns = ['data']
    # print(wordseg(data['data']))
    sentences = data.apply(lambda row: wordseg(row['data']),axis=1)
    # print(sentences)
    coln = ['token'+str(x) for x in sentences.columns.tolist()]
    # print(sentences)
    sentences.columns = coln
    sentences.to_pickle(PATH+outputFilename)
    print(sentences)
    return sentences

#%%
if __name__ == '__main__':

    # sentences = readRandomText(inputFilename='london.txt',outputFilename='real_data_london.pkl')
    kmer_list = load_seqs('../data/test.fa',3)
    print(kmer_list)
