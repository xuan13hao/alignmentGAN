# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:14:08 2023
"""
import sys
import torch

def decode(batch_size=1):
    model = torch.load('generator_3_16_101.pkl')
    out = model.sample(batch_size)
    dict = {}
    dict[1] = 'A'
    dict[2] = 'C'
    dict[3] = 'G'
    dict[4] = 'T'
    dict["A"] = 1
    dict["C"] = 2
    dict["G"] = 3
    dict["T"] = 4
    l = []
    for i in out:
        for n in i:
            l.append(dict[int(n)])
        print(''.join(l)+'\n')

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    
    decode(batch_size)
    # print(result[0])
    