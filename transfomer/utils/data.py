import torch
import pandas as pd
import numpy as np
import jieba
import re
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from .vocab import Vocab
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


#去除停用词，返回去除停用词后的文本列表
def clean_stopwords(contents):
    stopwords = {}.fromkeys([line.rstrip() for line in open('/home/idal-01/neu_cx/wcx_/transfomer/data/hit_stopwords.txt', encoding="utf-8")]) #读取停用词表
    stopwords_list = set(stopwords)
    words = [word for word in contents if word not in stopwords_list] #循环去除停用词
    sentence=''.join(words)   #去除停用词后组成新的句子

    return words

def get_tokens():
    path = "/home/idal-01/neu_cx/wcx_/transfomer/data/Train.csv"
    pd_all = pd.read_csv(path)

    print('总体数目：%d' % pd_all.shape[0])
    

    tokens = []
    for i in tqdm(range(pd_all.shape[0])):
        label = pd_all.Labels[i]
        data = pd_all.Text[i].replace(" ", "")
        
        # print(data)
        #去除“@用户名”部分的内容
        # 定义正则表达式
        pattern = re.compile(r'@(\S*)\s')
        # 执行替换操作
        data = pattern.sub('', data)

        # 中文文本  去除特殊字符
        data = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', data)
        # 分词
        words = jieba.lcut(data, use_paddle=True)
        tokens.extend(words)
    print(len(tokens))

    tokens = clean_stopwords(tokens)
    return tokens


def collate_fn(examples):
    
    inputs = [ex["ids"] for ex in examples]
    targets = [ex["targets"] for ex in examples]
    # 对batch中样本数据进行padding，使其长度相同
    max_len = max([len(seq) for seq in inputs])
    min_len = 5 #最短长度
    if max_len < min_len:
        # 如果最长序列的长度小于最小长度，进行填充
        # inputs = [F.pad(seq, (0, min_len - len(seq))) for seq in inputs]
        inputs[0] = F.pad(inputs[0], (1, min_len - len(inputs[0])), value=1)

    # 对序列进行填充
    inputs = pad_sequence(inputs,batch_first=True,padding_value=1)
    targets = torch.stack(targets, dim=0)
    return inputs , targets



class MYDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data =  data
    # 数据集大小
    def __len__(self):
        return len(self.data)
    # 根据索引获取对应数据
    def __getitem__(self, i):
        return self.data[i]

# import os
# import pickle
# if not os.path.exists("/home/idal-01/neu_cx/wcx_/transfomer/save/vocab"):
#     # 获取所有词
#     tokens = get_tokens()
#     # 建立字典
#     vocab = Vocab(tokens)
#     pickle.dump(vocab,open("/home/idal-01/neu_cx/wcx_/transfomer/save/vocab", 'wb'))
# else:
#     vocab = pickle.load(open("/home/idal-01/neu_cx/wcx_/transfomer/save/vocab", 'rb'))

# print(len(vocab))


