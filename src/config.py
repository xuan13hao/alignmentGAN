# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:52:48 2018
@author: natnij

Based on SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient,
    Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.
    Paper available here: https://arxiv.org/abs/1609.05473
Translated from the original tensorflow repo:
    https://github.com/LantaoYu/SeqGAN, and adjusted for wider usability.
Many thanks to the original authors.
"""
import torch
import os
from datetime import datetime

PATH = '../data/'
MAXINT = 10000
SEQ_LENGTH = 151# x: 'START' + tokens; y: tokens + 'END' 200 - 3 + 1 + 3
EMB_SIZE = 32
GENERATE_NUM = 1
FILTER_SIZE = list(range(1,SEQ_LENGTH))
# print(FILTER_SIZE)
# NUM_FILTER =  ([100] + [200] * 9 + [160] * SEQ_LENGTH)[0:SEQ_LENGTH-1]
NUM_FILTER =  ([1] + [1] * 9 + [1] * SEQ_LENGTH)[0:SEQ_LENGTH-1]
# print(NUM_FILTER)
DIS_NUM_EPOCH = 1
DIS_NUM_EPOCH_PRETRAIN = 3
GEN_NUM_EPOCH = 3
GEN_NUM_EPOCH_PRETRAIN = 3
GEN_HIDDEN_DIM = 48
ROLLOUT_ITER = 3
TOTAL_BATCH = 3

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NrGPU = 0

def openLog(filename='record.txt'):
    if os.path.exists(PATH+filename):
        append_write = 'a'
    else:
        append_write = 'w'
    log = open(PATH+filename, append_write)
    return log

if DEVICE.type == 'cuda':
    NrGPU = torch.cuda.device_count()
    print('number of GPUs available:{}'.format(NrGPU))
    log = openLog('gpu.txt')
    log.write('datetime:{}, device name:{}\n'.format(datetime.now(),
                                          torch.cuda.get_device_name(0)))
    log.write('Memory Usage:')
    log.write('\nAllocated:'+str(round(torch.cuda.memory_allocated(0)/1024**3,1))+'GB')
    log.write('\nCached:   '+str(round(torch.cuda.memory_cached(0)/1024**3,1))+'GB')
    log.close()
