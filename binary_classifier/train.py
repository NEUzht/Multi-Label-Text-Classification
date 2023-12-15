# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
import torch.nn as nn

from utils.dataset import *
from models.model import *
from utils.loss import *
from utils.scheduler import *
from utils.draw import *
from utils.stats import *


device = 'cuda:1' if cuda.is_available() else 'cpu'
df = pd.read_csv("Train.csv")

df['Labels'] = df['Labels'].apply(str)
# df['Text'] = df['Text'].str.replace(' ', '')

new_df = df[['Text','Labels']].copy()


# 构建class_to_id字典
class2id = get_class_to_id(df)
# 构建id_to_class字典
id2class = {v: k for k, v in class2id.items()}

print(class2id,"\n",id2class)
with open("./save/class2id.pkl", 'wb') as f:
    pickle.dump(class2id, f)
with open('./save/id2class.pkl', 'wb') as f:
    pickle.dump(id2class, f)



max_len = 256
train_batch_size = 64
valid_batch_size = 64
epochs = 5
learning_rate = 2e-5

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

data_list = new_df.values.tolist()

num_of_class,labels_nums_dict = stats(data_list, class2id)
print(f"num_of_class: {num_of_class}")
# data_list = filter_list(data_list, labels_nums_dict)
# data_list = data_list[0:100]
# 设置随机数种子以确保可复现性
random.seed(42)

# 随机化数据集的索引
indices = list(range(len(data_list)))
random.shuffle(indices)

# 计算训练集和测试集的划分点
train_size = 0.8  # 80% 的数据作为训练集
split_point = int(len(indices) * train_size)

# 划分数据集
train_indices = indices[:split_point]
test_indices = indices[split_point:]



# 构建训练集和测试集
train_dataset = [data_list[i] for i in train_indices]
test_dataset = [data_list[i] for i in test_indices]

print("FULL Dataset: {}".format(len(data_list)))
print("TRAIN Dataset: {}".format(len(train_dataset)))
print("TEST Dataset: {}".format(len(test_dataset)))


training_set = CustomDataset(train_dataset, tokenizer, max_len, class2id, id2class,train = True,augment=True) 
testing_set = CustomDataset(test_dataset, tokenizer, max_len, class2id, id2class,train = True)


train_params = {'batch_size': train_batch_size,
                'shuffle': True,
                'num_workers': 4
                }

test_params = {'batch_size': valid_batch_size,
                'shuffle': True,
                'num_workers': 4
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

weights = [v for k ,v in num_of_class.items()]
weights = [(len(data_list) - x) / len(data_list) for x in weights]

print(weights)

def train(index):

    model = BertClassifier(num_classes = len(id2class))
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, learning_rate * 0.1)

    # 画图记录训练过程
    animator = Animator(xlabel='epoch', xlim=[1, epochs], legend=['train_loss', 'train_acc','val_loss','val_acc'])
    animator_lr = Animator(xlabel='epoch', xlim=[1, epochs], legend=['learning_rate'])
    best_accuracy = 0.0
    save_path = '.pth/'
    
    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        acc = 0.0
        # fin_targets=[]
        # fin_outputs=[]
        train_qbar = tqdm(training_loader,desc="Training",total = len(training_loader))
        for step, data in enumerate(train_qbar, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            # print(targets,"\n",outputs)
            
            loss = BinaryLoss(outputs, targets,weights = weights,index = index)

            target = targets[:,index].unsqueeze(1)
            output = outputs[:,index].unsqueeze(1)
            # loss = nn.BCELoss()(output, target)
            # print(target,"\n",output)
            acc += (target == (output > 0.5).float()).sum().item()
            # fin_targets.extend(targets.cpu().detach().numpy().tolist())
            # fin_outputs.extend(torch.cat(outputs, dim=1).cpu().detach().numpy().tolist())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_qbar.set_postfix({'Loss':total_loss / (step + 1)},refresh=True)
        train_loss = total_loss/len(training_loader)
        train_acc = acc / len(train_dataset)
        # fin_outputs = np.array(fin_outputs) >= threshold
        # train_acc = metrics.accuracy_score(fin_targets, fin_outputs)
        #train_f1 = 
        print(f'Epoch: {epoch},train_acc: {train_acc:.2f}, train_loss:  {train_loss:.2f}')

        animator_lr.add(epoch+1, (optimizer.param_groups[0]['lr']))
        animator_lr.fig.savefig("./save/{}_learning_rate.jpg".format(id2class[index]), dpi=500, bbox_inches='tight')
        scheduler.step()


        model.eval()
        total_loss = 0
        acc = 0.0
        # fin_targets=[]
        # fin_outputs=[]
        with torch.no_grad():
            test_qbar = tqdm(testing_loader,desc="validing",total = len(testing_loader))
            for _, data in enumerate(test_qbar, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)

            
                # loss = BinaryLoss(outputs, targets,weights = weights,index = index)
                target = targets[:,index].unsqueeze(1)
                output = outputs[:,index].unsqueeze(1)
                loss = nn.BCELoss()(output, target)
                acc += (target == (output > 0.5).float()).sum().item()

                total_loss += loss.item()
                test_qbar.set_postfix({'Loss':total_loss / (step + 1)},refresh=True)
                
        val_loss = total_loss/len(testing_loader)
        val_acc  = acc / len(test_dataset)
        animator.add(epoch + 1, ( train_loss ,train_acc , val_loss, val_acc ))
        animator.fig.savefig(f".save/{id2class[index]}_loss.jpg", dpi=500, bbox_inches='tight')
        print(f"val_loss  = {val_loss:.2f}")
        print(f"Accuracy  = {val_acc:.2f}")
        # print(f"F1 Score (Micro) = {f1_score_micro:.2f}")
        # print(f"F1 Score (Macro) = {f1_score_macro:.2f}")
        if val_acc > best_accuracy:
        #  保存模型的状态字典，而不是整个模型
            torch.save(model.state_dict(),save_path +f"{id2class[index]}.pth")
            best_accuracy = val_acc

if __name__ == "__main__":
    print(class2id)
    for key, value in class2id.items():
        print(f"trainging-------------------------{key}")
        train(int(value))
        
    