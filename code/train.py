from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import os
from itertools import product
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from itertools import chain
from collections import Counter
import Generator
import helpers
import parse_cigar
"""
Created on Thu March 1 11:14:08 2023
"""

CUDA = True
VOCAB_SIZE = 4 #
START_LETTER = 0
BATCH_SIZE = 16
MLE_TRAIN_EPOCHS = 100
REF_NEG_SAMPLES = 200
SEQ_LENGTH = 109
GEN_EMBEDDING_DIM = 150 # 512
GEN_HIDDEN_DIM = 512 # 768

PATH = ""


def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        # print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, REF_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()
        total_loss = total_loss / ceil(REF_NEG_SAMPLES / float(BATCH_SIZE)) / SEQ_LENGTH

        print(' Average Train NLL = %.4f' % (total_loss))

# MAIN
if __name__ == '__main__':
    filename = "test.txt"
    cigar = parse_cigar.cigar_symbols_lists(filename,SEQ_LENGTH)
    gen = Generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, SEQ_LENGTH, gpu=CUDA)
    # print(cigar)
    if CUDA:
        gen = gen.cuda()
        cigar = cigar.cuda()
        
    # # PRETRAIN DISCRIMINATOR
    
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3)
    file_path = "pretrained_gen_edits.pkl"
    if os.path.isfile(file_path):
        print("File exists")
        gen = torch.load('pretrained_gen_edits.pkl')
    else:
        print("File does not exist")
        train_generator_MLE(gen, gen_optimizer, cigar, MLE_TRAIN_EPOCHS)
    # 8888888888888
        torch.save(gen, "pretrained_gen_edits.pkl")