import torch
import pandas as pd
import numpy as np
import jieba
from sklearn import metrics
import re
import os
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.vocab import Vocab
from utils.data import get_tokens, clean_stopwords, collate_fn
from TextCNN import TextCNN
from TextRNN import TextRNN
from utils.dataset import *
from torch.utils.data import Dataset, DataLoader
from utils.data import MYDataset
from torch import nn
import pickle
from torch import optim
from transformer import TransformerClassifier
from utils.utils import seeding, create_dir, Animator
from my_model import My_model
#超参数设置
#TextCNN
kernel_num = 1
kernel_size = [2,3,4]
#TextRNN
hidden_size = 256
layer_num  = 3
#transformer
num_heads = 4
d_hid = 1024
nlayers = 4

dropout = 0.5
class_num = 8

embed_dim = 512
num_epochs = 50
learing_rate = 0.0001
batch_size = 512
model_name = "Transformer"


if __name__ == "__main__":
    # 固定种子
    seeding(42)

    """建立字典"""
    if not os.path.exists("./save/vocab"):
        # 获取所有词
        tokens = get_tokens()
        # 建立字典
        vocab = Vocab(tokens)
        pickle.dump(vocab,open("./save/vocab", 'wb'))
    else:
        vocab = pickle.load(open("./save/vocab", 'rb'))

    print(len(vocab))
    for i, j in vocab.token_to_idx.items():
        print(i, j)
        if j > 5:
            break


    """加载数据"""
    if not os.path.exists("./save/train_data"):
        path = "./data/Train.csv"
        # 构建class_to_id字典
        
        pd_all = pd.read_csv(path)
        class2id = get_class_to_id(pd_all)
        # 构建id_to_class字典
        id2class = {v: k for k, v in class2id.items()}
        with open("./save/class2id.pkl", 'wb') as f:
            pickle.dump(class2id, f)
        with open("./save/id2class.pkl", 'wb') as f:
            pickle.dump(id2class, f)
        print(class2id,"\n",id2class)
        all_data = []
        for i in tqdm(range(pd_all.shape[0])):
            label = pd_all.Labels[i]
            data = pd_all.Text[i]
            # data = pd_all.Text[i].replace(" ", "")
            # #去除“@用户名”部分的内容
            # # 定义正则表达式
            # pattern = re.compile(r'@(\S*)\s')
            # # 执行替换操作
            # data = pattern.sub('', data)
            # # 中文文本  去除特殊字符
            # data = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', data)
            # # 分词
            # data = jieba.lcut(data, use_paddle=True)
            # print(data)
            # data = clean_stopwords(data)
            
            # data = vocab.to_idx(data)
            all_data.append([data,label])

        # 随机化数据集的索引
        print(len(all_data))
        indices = list(range(len(all_data)))
        random.shuffle(indices)

        # 计算训练集和测试集的划分点
        train_size = 0.8  # 80% 的数据作为训练集
        split_point = int(len(indices) * train_size)

        # 划分数据集
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        # 构建训练集和测试集
        train_data = [all_data[i] for i in train_indices]
        test_data = [all_data[i] for i in test_indices]

        torch.save(train_data,open("./save/train_data", 'wb'))
        torch.save(test_data,open("./save/test_data", 'wb'))
    else:
        with open("./save/class2id.pkl", 'rb') as f:
            class2id = pickle.load(f)
        with open("./save/id2class.pkl", 'rb') as f:
            id2class = pickle.load(f)
        train_data = torch.load(open("./save/train_data", 'rb'))
        test_data = torch.load(open("./save/test_data", 'rb'))
    print("FULL Dataset: {}".format(len(train_data)+len(test_data)))
    print("TRAIN Dataset: {}".format(len(train_data)))
    print("TEST Dataset: {}".format(len(test_data)))
    
    

    training_set = CustomDataset(train_data, vocab, class2id, train = True) 
    testing_set = CustomDataset(test_data, vocab,class2id, train = True)
    # print(training_set[:10])
    # train_dataset = MYDataset(train_data)
    # test_dataset = MYDataset(test_data)
    train_data_loader = DataLoader(training_set,batch_size = batch_size,collate_fn=collate_fn,shuffle=True)
    test_data_loader = DataLoader(testing_set,batch_size=batch_size,collate_fn=collate_fn,shuffle=False)

    # 加载模型
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    if model_name == "TextCNN":
        # TextCNN
        model = TextCNN(kernel_num, kernel_size, len(vocab), embed_dim, dropout, class_num)
    elif model_name == "TextRNN":
        # TextRNN
        model = TextRNN(embed_dim, len(vocab), hidden_size, layer_num, device)
    elif model_name == "Transformer":
        # Transformer
        model = TransformerClassifier(len(vocab), embed_dim, num_heads, d_hid, nlayers)
    else:
        model = My_model(len(vocab), embed_dim)
    print("Trainning by model {}.".format(model_name))

    model.to(device)

    # 训练过程
    nll_loss = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(),learing_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, learing_rate * 0.1)

    # 画图记录训练过程
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train_loss', 'train_acc','val_loss','val_acc'])
    animator_lr = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['learning_rate'])

    """ 训练"""
    threshold = 0.5
    best_acc = float(0.0)
    for epoch in range(num_epochs):
        total_loss = 0
        acc = 0
        model.train()
        fin_targets=[]
        fin_outputs=[]
        for inputs,targets in tqdm(train_data_loader,desc=f"Training Epoch {epoch}"):
            inputs = inputs.to(device)
            targets= targets.to(device)
            # print(inputs, type(inputs), inputs.shape)
            # exit()
            y_hat = model(inputs)
            loss = nll_loss(y_hat,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(y_hat.cpu().detach().numpy().tolist())

        train_loss = total_loss/len(train_data_loader)
        fin_outputs = np.array(fin_outputs) >= threshold
        train_acc = metrics.accuracy_score(fin_targets, fin_outputs)
        print(f"Loss:{train_loss:.2f}")
        print(f"Acc:{train_acc:.2f}")


        animator_lr.add(epoch+1, (optimizer.param_groups[0]['lr']))
        animator_lr.fig.savefig("./save/learning_rate.jpg", dpi=500, bbox_inches='tight')

        scheduler.step()

        # 测试过程
        total_loss = 0
        acc = 0
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        for inputs,targets in tqdm(test_data_loader,desc=f"Testing"):
            inputs = inputs.to(device)
            targets= targets.to(device)
            with torch.no_grad():
                output = model(inputs)
                loss = nll_loss(output,targets)
                total_loss += loss.item()
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(output.cpu().detach().numpy().tolist())
        

        val_loss = total_loss/len(test_data_loader)
        fin_outputs = np.array(fin_outputs) >= threshold
        val_acc = metrics.accuracy_score(fin_targets, fin_outputs)
        # print(fin_targets, fin_outputs)
        f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
        f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
        
        animator.add(epoch + 1, ( train_loss, train_acc , val_loss, val_acc ))
        if model_name == "TextCNN":
            animator.fig.savefig(f"./save/{model_name}_{kernel_num}_{kernel_size}_{learing_rate}_{embed_dim}.jpg", dpi=500, bbox_inches='tight')
        elif model_name == "TextRNN":
            animator.fig.savefig(f"./save/{model_name}_{hidden_size}_{layer_num}_{learing_rate}_{embed_dim}.jpg", dpi=500, bbox_inches='tight')
        elif model_name == "Transformer":
            animator.fig.savefig(f"./save/{model_name}_{num_heads}_{d_hid}_{nlayers}_{learing_rate}_{embed_dim}.jpg", dpi=500, bbox_inches='tight')
        else:
            animator.fig.savefig(f"./save/{model_name}_{learing_rate}_{embed_dim}.jpg", dpi=500, bbox_inches='tight')
        # 输出在测试集上的准确率
        print(f"val_loss:{val_loss:.2f}")
        print(f"val_Acc:{val_acc}")


        if val_acc > best_acc:
            print(f"acc improved from {best_acc:2.4f} to {val_acc:2.4f}.")
            best_acc = val_acc
            if model_name == "TextCNN":
                torch.save(model.state_dict(),f'./pth/{model_name}__{kernel_num}_{kernel_size}_{embed_dim}_{epoch}_steps_{val_acc:.2f}.pt')
            elif model_name == "TextRNN":
                torch.save(model.state_dict(),f'./pth/{model_name}__{hidden_size}_{layer_num}_{embed_dim}_{epoch}_steps_{val_acc:.2f}.pt')
            elif model_name == "Transformer":
                torch.save(model.state_dict(),f'./pth/{model_name}__{num_heads}_{d_hid}_{nlayers}_{embed_dim}_{epoch}_steps_{val_acc:.2f}.pt')
            else:
                torch.save(model.state_dict(),f'./pth/{model_name}__{embed_dim}_{epoch}_steps_{val_acc}.pt')






