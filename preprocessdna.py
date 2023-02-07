from torch import nn
from gensim.models import Word2Vec
import torch
import os
import load_data
class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="w2v_all.model"): #首先定义类的一些属性
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前训练好的word to vec 模型读进来
        model = Word2Vec.load(self.w2v_path)
        # weights = torch.LongTensor(model.wv.vectors,requires_grad=True)
        # embedding = nn.Embedding.from_pretrained(weights)
        self.embedding = model
        # self.embedding = Word2Vec.load(self.w2v_path)
        # print(Word2Vec.load(self.w2v_path))
        self.embedding_dim = self.embedding.vector_size #embedding的维度就是训练好的Word2vec中向量的长度
    def add_embedding(self, word):
        # 把word（"<PAD>"或"<UNK>"）加进embedding，并赋予他一个随机生成的representation vector
        # 因为我们有时候要用到"<PAD>"或"<UNK>"，但它俩本身没法放到word2vec中训练而且它俩不需要生成一个能反应其与其他词关系的向量，故随机生成
        vector = torch.empty(1, self.embedding_dim)#生成空的
        torch.nn.init.uniform_(vector)#随机生成
        self.word2idx[word] = len(self.word2idx)#在word2idx放入对应的index
        self.idx2word.append(word)#在idx2word中放入对应的word
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)#在embedding_matrix中加入新的vector
    def make_embedding(self, load=True):
        # print("Get embedding ...")
        # 取得训练好的 Word2vec word embedding
        if load:
            print("loading DNA to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作一个 word2idx 的 dictionary
        # 制作一个 idx2word 的 list
        # 制作一个 word2vector 的 list
        # print(self.embedding.wv.key_to_index)
        for i, word in enumerate(self.embedding.wv.key_to_index):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['哈'] = 1 
            #e.g. self.index2word[1] = '哈'
            #e.g. self.vectors[1] = '哈' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        # print('embedding')
        # print(self.embedding_matrix)
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # print(self.embedding_matrix)
        # 將"<PAD>"和"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("Total words: {}".format(len(self.embedding_matrix)))
        # print(self.embedding_matrix)
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len: #多的直接截断
            sentence = sentence[:self.sen_len]
        else:                            #少的添加"<PAD>"
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子里面的字变成相对应的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
            print(sentence_list)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把labels转成tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)
if __name__ == "__main__":

    path_prefix = './'
    w2v_path = os.path.join(path_prefix, 'w2v_all.model')
    # train_x, y = load_training_data(train_with_label)
    # train_x_no_label = load_training_data(train_no_label)
    train_seq = load_data.get_seqs("train.fa")
    test_seq = load_data.get_seqs("test.fa")
    # print(seq_list)
    train_kmer_list = load_data.getKmerList(train_seq,3)
    test_kmer_list = load_data.getKmerList(test_seq,3)
    # 对input和labels做预处理
    seq_len = 148
    # train_preprocess = Preprocess(train_kmer_list, seq_len, w2v_path=w2v_path)
    # embedding = train_preprocess.make_embedding(load=True)
    # train_x = train_preprocess.sentence_word2idx()
    # print(train_x)
    test_preprocess = Preprocess(test_kmer_list, seq_len, w2v_path=w2v_path)
    embedding = test_preprocess.make_embedding(load=True)
    # test_x = test_preprocess.make_embedding_dict()
    # print(test_x)
    # y = preprocess.labels_to_tensor(y)
