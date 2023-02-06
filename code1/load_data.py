import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import fasta
from gensim.models import word2vec
path_prefix = './'

def load_training_data(path='train_data.txt'):
    # 把training时需要的data读入
    # 如果是'training_label.txt'，需要读取它的label，如果是'training_nolabel.txt'，不需要读取label（本身也没有label）
    if 'training_label' in path: #判断training_label这几个字在不在path中，以判断需不需要读取label
        #读入存在txt中文本数据的常用方式
        with open(path, 'r') as f: 
            lines = f.readlines() #一行一行读入数据
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines] #第二列之后是文本数据
        y = [line[0] for line in lines] #第一列是label
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data'):
    # 把testing时需要的data读进来
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels): #定义自己的评价函数，用分类的准确率来评价
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於0.5為有惡意
    outputs[outputs<0.5] = 0 # 小於0.5為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

# 这个block是用来训练word to vector 的 word embedding
# 注意！这个block在训练word to vector时是用cpu，可能要花到10分钟以上(我试了一下，确实是要很久)
def get_seqs(input_file):
	seq_reads = []
	for record in fasta.parse(input_file):
		seq_reads.append(str(record.seq))
	return seq_reads

def DNAToWord(dna, K):
    length = len(dna)
    sub_seq = []
    for i in range(length - K + 1):
        sub_seq.append(dna[i: i + K])
    return sub_seq

def getKmerList(DNAdata):
    list1 = []
    for DNA in DNAdata:
        DNA = str(DNA).upper()
        k = len(DNA)
        list1.append(DNAToWord(DNA,k))
    return list1

def train_word2vec(x):
    # 训练word to vector 的 word embedding
    #size是神经网络的层数，window是窗口长度，min_count是用来忽略那些出现过少的词语，worker是线程数，iter是循环次数
    W = len(x[0][0])
    # print(len(x[0][0]))
    model = word2vec.Word2Vec(x, vector_size =200, window=W, min_count=1, workers=12, sg=1)
    return model

if __name__ == "__main__":
    train_seq = get_seqs("train.fa")
    test_seq = get_seqs("test.fa")
    # print(seq_list)
    train_kmer_list = getKmerList(train_seq)
    test_kmer_list = getKmerList(test_seq)
    # print(train_kmer_list)
    # print("loading training data ...")
    # train_x = load_training_data('train_data.txt')
    # train_x_no_label = load_training_data('training_nolabel.txt')

    # print("loading testing data ...")
    # test_x = load_testing_data('test_data.txt')
    # print(train_x)
    # # print(y)
    # print(train_x_no_label)
    model = train_word2vec(train_kmer_list + test_kmer_list)
    # print(model)
    # print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(path_prefix, 'w2v_all.model')) #将模型保存这一步可以使得后续的训练更方便，是一个很好的习惯
