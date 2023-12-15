import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer,BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.1, bert_model_path='./bert-base-chinese'):
        super(BertClassifier, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(768, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids=None, mask=None, token_type_ids=None):
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        x = self.dropout(output.last_hidden_state)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class MultiBinaryClass(torch.nn.Module):
    def __init__(self,num_classes):
        super(MultiBinaryClass, self).__init__()
        self.num_classes = num_classes
        self.bert = transformers.BertModel.from_pretrained('./bert-base-chinese')
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()
            ) for _ in range(self.num_classes)
        ])

    def forward(self, ids=None, mask=None, token_type_ids=None):
        output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        pooled_output = output_1[1]
        outputs = [classifier(pooled_output) for classifier in self.classifiers]
        return outputs
    


# model = BertClassifier(8)
# print(model.num_classes)