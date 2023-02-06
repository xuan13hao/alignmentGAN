from torch import nn
from gensim.models import Word2Vec
import torch
import os
import sys
import load_data
import preprocessdna 
import discriminator_kmer
import generator_kmer
import torch.optim as optim
import helpers

from math import ceil
import torch.optim as optim
import torch.nn as nn


CUDA = False
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 1
START_LETTER = 0
BATCH_SIZE = 1
MLE_TRAIN_EPOCHS = 2
ADV_TRAIN_EPOCHS = 2
POS_NEG_SAMPLES = 1

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    # print("oracle = ",oracle)
    # print("real_data_samples = ",real_data_samples)
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            # print(real_data_samples[i:i + BATCH_SIZE])
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            # print("loss = ",loss)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)
        # print(oracle_loss)
        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


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
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

    print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(1)
    neg_val = generator.sample(1)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))


if __name__ == "__main__":

    path_prefix = './'
    w2v_path = os.path.join(path_prefix, 'w2v_all.model')
    # train_x, y = load_training_data(train_with_label)
    # train_x_no_label = load_training_data(train_no_label)
    train_seq = load_data.get_seqs("train.fa")
    test_seq = load_data.get_seqs("test.fa")
    # print(seq_list)
    train_kmer_list = load_data.getKmerList(train_seq)
    test_kmer_list = load_data.getKmerList(test_seq)
    # 对input和labels做预处理
    seq_len = 1
    train_preprocess = preprocessdna.Preprocess(train_kmer_list, seq_len, w2v_path=w2v_path)
    # print(train_preprocess)
    real_embedding = train_preprocess.make_embedding(load=True)
    # print(embedding)
    train_real = train_preprocess.sentence_word2idx()
    # print(train_real)
    gen_preprocess = preprocessdna.Preprocess(test_kmer_list, seq_len, w2v_path=w2v_path)
    gen_embedding = gen_preprocess.make_embedding(load=True)
    # train_ref = train_preprocess.sentence_word2idx()

    gen_gener = generator_kmer.Generator(gen_embedding, GEN_HIDDEN_DIM, seq_len, gpu=CUDA)
    gen = generator_kmer.Generator(gen_embedding, GEN_HIDDEN_DIM, seq_len, gpu=CUDA)
    dis = discriminator_kmer.Discriminator(real_embedding, DIS_HIDDEN_DIM, seq_len, gpu=CUDA)

    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    
    train_generator_MLE(gen, gen_optimizer, gen_gener, real_embedding, MLE_TRAIN_EPOCHS)
    dis_optimizer = optim.Adagrad(dis.parameters())
    #real_embedding
    train_discriminator(dis, dis_optimizer, real_embedding, gen, gen_gener, 2, 2)
    



    # print(gen_optimizer)


    # if CUDA:
    #     oracle = ori.cuda()
    #     gen = gen.cuda()
    #     dis = dis.cuda()