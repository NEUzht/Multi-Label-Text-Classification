import torch
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from utils.dataset import *
from models.model import *
from utils.loss import *
from utils.scheduler import *
from utils.draw import *
from utils.stats import *

def predict(model, predict_loader, model_path, class2id, id2class):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = BERTClass(num_classes = len(id2class))
    
    all_result = []
    for _, value in class2id.items():
        model_dict = torch.load(model_path / (id2class[value] + ".pth"))
        model.load_state_dict(model_dict)
        model.to(device)
        model.eval()
        with torch.no_grad():
            predict_bar = tqdm(predict_loader,desc= id2class[value],total = len(predict_loader))
            result = []
            for step, data in enumerate(predict_bar, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                outputs = model(ids, mask, token_type_ids)
                output = outputs[:,value].unsqueeze(1)
                result.append(output > 0.5)
            all_result.append(torch.vstack(result))
    all_result = torch.hstack(all_result).cpu().numpy().tolist()
    label_list = []
    for labels in all_result:
        temp = []
        for index in range(len(labels)):
            if labels[index] == True:
                temp.append(id2class[index])
        
        label_list.append(str(temp))
    index_ = []
    labels = []
    for index, data in enumerate(label_list):
        index_.append(index + 1)
        labels.append(str(data))
    # 将列表转为字典，然后创建 DataFrame
    data_dict = {"ID": index_, "Labels": labels}
    df = pd.DataFrame(data_dict)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv("./data/predict_zhang.csv", index=False)

    

if __name__ == "__main__" :

    print("testing................")
    with open('./save/class2id.pkl', 'rb') as f:
        class2id = pickle.load(f)
    with open('./save/id2class.pkl', 'rb') as f:
        id2class = pickle.load(f)
    
    path = "/home/idal-01/neu_cx/wcx_/data/test_zhang.csv"
    model_path = Path("./pth")
    df = pd.read_csv(path, encoding='utf-8')
    test_dataset = df['Text'].copy().values.tolist()
    tokenizer = BertTokenizer.from_pretrained('/home/idal-01/neu_cx/wcx_/bert-base-chinese')
    test_dataset = CustomDataset(test_dataset, tokenizer, 256, class2id, id2class,train = False)
    test_params = {'batch_size': 32,
                    'shuffle': False,
                    'num_workers': 4
                    }
    predict_loader = DataLoader(test_dataset, **test_params)
    
    model = BERTClass(num_classes = len(id2class))
    predict(model, predict_loader, model_path, class2id, id2class)
    

    
    print("predict finish!")