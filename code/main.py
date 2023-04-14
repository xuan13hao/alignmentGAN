from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
from itertools import product
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from itertools import chain
from collections import Counter
import Generator
import Discriminator
import helpers
"""
Created on Thu March 1 11:14:08 2023
"""

CUDA = True
VOCAB_SIZE = 65 # 4097 16,384
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 30
POS_NEG_SAMPLES = 369
SEQ_LENGTH = 99
GEN_EMBEDDING_DIM = 128 # 512
GEN_HIDDEN_DIM = 768 # 768
DIS_EMBEDDING_DIM = 128 # 512
DIS_HIDDEN_DIM = 128 # 768

# oracle_samples_path = './oracle_samples.trc'
# oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
# pretrained_gen_path = './gen_MLEtrain.pkl'
# pretrained_dis_path = './dis_pretrain.pkl'
PATH = ""

def generate_all_kmers(k):
    alphabet = "ACGT"
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    # print(kmers)
    idx = 0
    kmer_dict = {}
    for kmer in kmers:
        idx = idx + 1
        kmer_dict[kmer] = idx
    return kmer_dict

def read_sampleFile(k, file='kmer.pkl', pad_token='PAD',num=None):
    if file[-3:]=='pkl' or file[-3:]=='csv':
        if file[-3:] == 'pkl':
            data = pd.read_pickle(file)
        else:
            data = pd.read_csv(file)
        # print(data)
        if num is not None:
            num = min(num,len(data))
            data = data[0:num]
        lineList_all = data.values.tolist()
        # print(lineList_all)
        characters = set(chain.from_iterable(lineList_all))
        # print(characters)
        lineList_all = [w for w in lineList_all]
        # print(lineList_all)
        x_lengths = [len(x) - Counter(x)[pad_token] for x in lineList_all]
        # print(x_lengths)
    else:
        lineList_all = list()
        characters = list()
        x_lengths = list()
        count = 0
        with open(file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line.strip()
                lineList = list(line)
                try:
                    lineList.remove('\n')
                except ValueError:
                    pass
                x_lengths.append(len(lineList) + 1)
                
                characters.extend(lineList)
                # print(characters)
                # print(len(characters))
                if len(lineList)<SEQ_LENGTH:
                    lineList.extend([pad_token] * (SEQ_LENGTH - len(lineList)))
                lineList_all.append(lineList)
                count += 1
                if num is not None and count >= num:
                    break
    vocabulary = generate_all_kmers(k)
    tmp = sorted(zip(x_lengths,lineList_all), reverse=True)
    x_lengths = [x for x,y in tmp]
    lineList_all = [y for x,y in tmp]
    generated_data = [int(vocabulary[x]) for y in lineList_all for i,x in enumerate(y) if i<SEQ_LENGTH]
    # to tensor
    x = torch.tensor(generated_data).view(-1,SEQ_LENGTH)
    # print(x)
    #x.int(), vocabulary, reverse_vocab, x_lengths
    return x.int(), vocabulary, x_lengths


def train_baseline(oracle,oracle_opt,train_data,epochs):
# Define your loss function
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        for i in range(BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(train_data[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            oracle_opt.zero_grad()
            loss = oracle.batchNLLLoss(inp, target)
            loss.backward()
            oracle_opt.step()
            total_loss += loss.data.item()
        # total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN
        # print(' average_train_NLL = %.4f' % (total_loss))

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        # print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

        #     if (i / BATCH_SIZE) % ceil(
        #                     ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
        #         print('.', end='')
        #         sys.stdout.flush()

        # # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / SEQ_LENGTH

        # # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)

        # print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
        print(' Average Train NLL = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        # print("pg_loss:", print)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, SEQ_LENGTH,
                                                   start_letter=START_LETTER, gpu=CUDA)

    print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    # pos_val = oracle.sample(50)
    # neg_val = generator.sample(50)
    # val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)#???????????
    # val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)#???????????
    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            # print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                # print("BECloss = ",loss)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

            #     if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
            #             BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
            #         print('.', end='')
            #         sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            # val_pred = discriminator.batchClassify(val_inp)
            # print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
            #     total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/101.))
            print(' Average Loss = %.4f, Train acc = %.4f' % (
                total_loss, total_acc))
            

# MAIN
if __name__ == '__main__':
    real, vocabulary, sentence_lengths = read_sampleFile(file = "kmer.pkl",k = 3)
    fake, vocabulary_ref, sentence_lengths_ref = read_sampleFile(file = "reference.pkl", k = 3)
    oracle = Generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, SEQ_LENGTH, gpu=CUDA)
    orcale_optimizer = optim.Adam(oracle.parameters(), lr=1e-3)
    print("Real Sample num = ",len(real))
    print("Fake Sample num = ",len(fake))
    # print(vocabulary)
    
    # train_baseline(oracle,orcale_optimizer,x,MLE_TRAIN_EPOCHS)
    # torch.save(oracle, 'oracle.pkl')
    # oracle.torch.load("oracle_state_dict_path")
    # print(oracle)
    # oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)

    x_ref = fake
    # train_baseline(oracle,x,MAX_SEQ_LEN)
    oracle_samples = real
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()
    # print(oracle_samples)
    # print(x)
    gen = Generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, SEQ_LENGTH, gpu=CUDA)
    dis = Discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, SEQ_LENGTH, gpu=CUDA)

    if CUDA:
        print("Running CUDA")
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        real = real.cuda()
        fake = fake.cuda()

    # GENERATOR MLE TRAINING use reference to train generator
    print('Starting Generator MLE Training...')
    # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3)
    train_generator_MLE(gen, gen_optimizer, oracle, fake, MLE_TRAIN_EPOCHS)
    # 8888888888888
    # torch.save(gen, pretrained_gen_path)
    # torch.save(gen_optimizer, "opt_pretrained_gen_MLE.pkl")
    # # torch.save(gen.state_dict(), pretrained_gen_path)
    # # gen.load_state_dict(torch.load(pretrained_gen_path))
    # gen = torch.load(pretrained_gen_path)4rf
    # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    # # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters(),lr=1e-3)
    train_discriminator(dis, dis_optimizer, real, gen, oracle, 50, 3)
    # torch.save(dis, pretrained_dis_path)
    # torch.save(dis_optimizer, "opt_pretrained_dis.pkl")
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    # print('\nStarting Adversarial Training...')
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                            start_letter=START_LETTER, gpu=CUDA)
    # print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        # train_generator_MLE(gen, gen_optimizer, oracle, fake, 1)
        
        train_generator_PG(gen, gen_optimizer, oracle, dis, 2)
        
        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, real, gen, oracle, 5, 3)
    try:
        torch.save(gen, 'generator.pkl')
        torch.save(dis, 'discriminator.pkl')
        print('successfully saved generator model.')
    except:
        print('error: model saving failed!!!!!!')