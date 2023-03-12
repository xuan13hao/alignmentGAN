# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:14:08 2018
"""
import sys
import torch
def decode(token_tbl, reverse_vocab, log=None):
    words_all = []
    for n in token_tbl:
        words = [reverse_vocab[int(l)] for l in n]
        words_all.append(words[1:])
        if log is not None:
            print(''.join(words[1:])+'\n')
    return words_all

def main(batch_size=1):
    model = torch.load('generator.pkl')
    out = model.sample(1)
    dict = {}
    dict[0] = "START"
    dict[1] = 'A'
    dict[2] = 'C'
    dict[3] = 'G'
    dict[4] = 'T'

    dict["START"] = 0
    dict["A"] = 1
    dict["C"] = 2
    dict["G"] = 3
    dict["T"] = 4
    l = []
    for i in out:
        for n in i:
            l.append(dict[int(n)])
    print(''.join(l[1:])+'\n')
    # reverse_vocab = torch.load('reference.pkl')
    # log = []
    # num = model.generate(batch_size=batch_size)
    # result = decode(num, reverse_vocab, log)
    # return result

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    
    main(batch_size)
    # print(result[0])
    