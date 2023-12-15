import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from .augmentation import *
def get_class_to_id(df):
    # 获取唯一的标签列表
    unique_labels = set()

    # 遍历DataFrame中的标签列
    for index, row in df.iterrows():
        labels = row['Labels']

        for label in eval(labels):
            unique_labels.add(label)
    sorted_list = sorted(unique_labels)
    class_to_id = {label: idx for idx, label in enumerate(sorted_list)}

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

    def __init__(self, data_list, tokenizer, max_len, class2id, id2class,train,augment = False):
        self.class2id =  class2id
        self.id2class = id2class
        self.tokenizer = tokenizer
        self.data = data_list
        self.train =  train
        self.augment = augment
        self.comment_text = [item[0] for item in self.data]
        if self.train:
            self.targets = [eval(item[1]) for item in self.data]
        # print(self.targets)
        self.max_len = max_len
        

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        # print(comment_text)
        comment_text = " ".join(comment_text.split())
        

        # comment_text = comment_text.replace(" ","")
        # print(comment_text)
        # # print(len(comment_text))
        # print(self.tokenizer.tokenize(comment_text))
        # print(type(self.tokenizer.tokenize(comment_text)))
        if self.augment:
            probability = random.uniform(0, 1)
            # print(probability)
            if probability <= 0.2:
                token_list = swap_token(self.tokenizer.tokenize(comment_text))
                comment_text = ' '.join(token_list)
            elif 0.2 < probability <= 0.4:
                token_list = dele_tokens(self.tokenizer.tokenize(comment_text))
                comment_text = ' '.join(token_list)
            elif 0.4 < probability <= 0.6:
                token_list = insert_punctuations(self.tokenizer.tokenize(comment_text))
                comment_text = ' '.join(token_list)
        # 是否加入'[CLS]'以及'[SEP]'
        # comment_text = '[CLS]' + comment_text + '[SEP]'
        
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        if self.train:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(list_id_to_class(self.targets[index],self.class2id), dtype=torch.float)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            }


    
            