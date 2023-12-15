import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from .augmentation import *
import re
import jieba
from .data import *
def get_class_to_id(df):
    # 获取唯一的标签列表
    unique_labels = set()

    # 遍历DataFrame中的标签列
    for index, row in df.iterrows():
        labels = row['Labels']

        for label in eval(labels):
            unique_labels.add(label)
    
    class_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    return class_to_id

def list_id_to_class(target_list, class2id):
        if isinstance(target_list[0], list):
            target_id_list = [[0]*len(class2id) for _ in range(len(target_list))]
        if isinstance(target_list[0],str):
            target_id_list = [0]*len(class2id)
        # print(str)
        for index,target in enumerate(target_list):
            if isinstance(target, list):
                for str_ in target:
                    target_id_list[index][class2id[str_]] = 1;
            if isinstance(target,str):
                # print(target,type(target))
                target_id_list[class2id[target]] = 1;
        
        return target_id_list;


class CustomDataset(Dataset):

    def __init__(self, data_list,  vocab, class2id, train, augment = False):
        self.class2id =  class2id
        self.data = data_list
        self.train =  train
        self.vocab = vocab
        self.augment = augment
        self.comment_text = [item[0] for item in self.data]
        if self.train:
            self.targets = [eval(item[1]) for item in self.data]
        # print(self.targets)
        

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        data = str(self.comment_text[index])
        data = data.replace(" ", "")

        #去除“@用户名”部分的内容
        # 定义正则表达式
        pattern = re.compile(r'@(\S*)\s')
        # 执行替换操作
        data = pattern.sub('', data)
        # 中文文本  去除特殊字符
        data = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', data)
        # 分词
        data = jieba.lcut(data, use_paddle=True)
        data = clean_stopwords(data)
        if self.augment:
            probability = random.uniform(0, 1)
            # print(probability)
            if probability <= 0.2:
                token_list = swap_token(data)
                data = ' '.join(token_list)
            elif 0.2 < probability <= 0.4:
                token_list = dele_tokens(data)
                data = ' '.join(token_list)
            # elif 0.4 < probability <= 0.6:
            #     token_list = insert_punctuations(data)
            #     data = ' '.join(token_list)
        data = self.vocab.to_idx(data)
        
        
    
        
        # 是否加入'[CLS]'以及'[SEP]'
        # comment_text = '[CLS]' + comment_text + '[SEP]'
        
        if self.train:
            return {
                'ids': torch.tensor(data, dtype=torch.long),
                'targets': torch.tensor(list_id_to_class(self.targets[index],self.class2id), dtype=torch.float)
            }
        else:
            return {
                'ids': torch.tensor(data, dtype=torch.long),
            }


    
            