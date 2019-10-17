# coding = utf-8

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

def train_word_vec(data_path, word_vec_path):
    '''
    读取文件，训练并获得词向量
    
    Args:
        data_path str 文本数据路径
        word_vec_path str 训练的词向量存储路径
    '''
    model = Word2Vec(LineSentence(data_path), size = 512, window = 5, min_count = 1, sg = 1)
    model.save(word_vec_path)
    return model

def get_word_vec(word_vec_path):
    '''
    读取预训练好的词向量
    Args:
        word_vec_path str 词向量路径
    Return
        model array 预训练好的词向量
    '''
    return Word2Vec.load(word_vec_path)

def preprocess(data_path, save_path):
    save_f = open(save_path, 'w', encoding='utf-8')
    with open(data_path, encoding='utf-8') as f:
        for s in f.readlines():
            s = " ".join(list(s.replace(' ', '')))
            save_f.write(s)

if __name__ == '__main__':
    # preprocess('./wiki_zh.txt', './char_zh.txt')
    train_word_vec('./char_zh.txt', './w2v.model')
    model = get_word_vec('./w2v.model')
    print(model.most_similar('书'))