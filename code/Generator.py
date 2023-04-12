"""
Created on Thu Feb 20 11:14:08 2023
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
# from data_processing import read_sampleFile


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        # x, _, reverse_vocab, _ = read_sampleFile(file = "reference.pkl")
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu
        self.num_layers = 12
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=self.num_layers)
        # self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        #h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        h = autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch
    # def batchPGLoss(self, inp, target, reward):
    #     batch_size, seq_len = inp.size()
    #     inp = inp.permute(1, 0)
    #     target = target.permute(1, 0)
    #     h = self.init_hidden(batch_size)
        
    #     loss = 0
    #     for i in range(seq_len):
    #         out, h = self.forward(inp[i], h)
    #         log_probs = F.log_softmax(out, dim=-1)  # 计算对数概率
    #         loss += -(log_probs[range(batch_size), target.data[i]] * reward).mean()  # Compute KL loss
    #     return loss


    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """
        # print("reward = ",reward)
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q 
        # print("pgloss = ",loss/batch_size)
        return loss/batch_size

'''
def batchPGLoss(self, inp, target, reward):
    """
    Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
    Uses sequence similarity as a reward.

    Inputs: inp, target
        - inp: batch_size x seq_len
        - target: batch_size x seq_len
        - reward: batch_size x seq_len (similarity score for each token of each sentence in the batch)

        inp should be target with <s> (start letter) prepended
    """
    batch_size, seq_len = inp.size()
    inp = inp.permute(1, 0)          # seq_len x batch_size
    target = target.permute(1, 0)    # seq_len x batch_size
    h = self.init_hidden(batch_size)

    loss = 0
    for i in range(seq_len):
        out, h = self.forward(inp[i], h)
        # TODO: should h be detached from graph (.detach())?
        for j in range(batch_size):
            # Use negative similarity score as reward (higher similarity = lower loss)
            similarity_reward = -reward[j][i]
            # Compute log softmax probabilities of the model output
            log_probs = F.log_softmax(out[j], dim=-1)
            # Use the target token as the index to select the corresponding log probability
            target_index = target.data[i][j]
            log_prob = log_probs[target_index]
            # Compute the weighted loss (log probability * similarity reward)
            weighted_loss = log_prob * similarity_reward
            loss += weighted_loss
    # Average the loss over the batch size and sequence length
    return loss / (batch_size * seq_len)

'''


'''
    def batchNLLLoss(self, inp, target, alpha, beta):
        """
        Returns the NLL Loss for predicting target sequence.
The batchNLLLoss() method now takes two additional hyperparameters alpha and beta, which control the relative importance of the negative log-likelihood loss and the similarity-based loss, respectively. The function first initializes the two loss functions: nll_loss_fn for the negative log-likelihood loss and cos_similarity_fn for the cosine similarity-based loss.

Then, the function iterates over the sequence length and computes the two loss terms for each sequence. The negative log-likelihood loss is computed as before using nll_loss_fn, while the similarity-based loss is computed using cos_similarity_fn as in the previous example.
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        hidden = self.init_hidden(batch_size)

        nll_loss_fn = nn.NLLLoss()
        cos_similarity_fn = cosine_similarity

        nll_loss = 0
        sim_loss = 0

        for i in range(seq_len):
            out, hidden = self.forward(inp[i], hidden)
            nll_loss += nll_loss_fn(out.view(-1, out.size(-1)), target[i].view(-1))

            # Compute similarity-based loss
            pred_seq = out.argmax(dim=-1).cpu().numpy()   # convert to numpy array for cosine_similarity
            target_seq = target[i].cpu().numpy()
            sim_loss -= torch.tensor(cos_similarity_fn(pred_seq, target_seq)).mean()

        # Combine losses with weights
        loss = alpha * nll_loss + beta * sim_loss

        return loss / seq_len  # per sequence

'''