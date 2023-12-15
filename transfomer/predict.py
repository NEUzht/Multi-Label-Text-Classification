import jieba
import pickle
import re
from data import clean_stopwords, collate_fn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer import TransformerClassifier
from TextCNN import TextCNN
from TextRNN import TextRNN
import torch.nn.functional as F

model_name = "Transformer"
#超参数设置
#TextCNN
kernel_num = 2
kernel_size = [2,3,4]
#TextRNN
hidden_size = 128
layer_num  = 2
#transformer
num_heads = 4
d_hid = 512
nlayers = 4

dropout = 0.2
class_num = 2
embed_dim = 128
vocab = pickle.load(open("./save/vocab", 'rb'))


# 加载模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

if model_name == "TextCNN":
    # TextCNN
    model = TextCNN(kernel_num, kernel_size, len(vocab), embed_dim, dropout, class_num)
elif model_name == "TextRNN":
    # TextRNN
    model = TextRNN(embed_dim, len(vocab), 64, 2)
else:
    # Transformer
    model = TransformerClassifier(len(vocab), embed_dim, num_heads, d_hid, nlayers)
print("Trainning by model {}.".format(model_name))

model.to(device)
model.load_state_dict(torch.load("./model/Transformer__4_512_4_128_18_steps_0.91.pt"))
model.eval()



def predict(data: str):
    #去除“@用户名”部分的内容
    # 定义正则表达式
    pattern = re.compile(r'@(\S*)\s')
    # 执行替换操作
    data = pattern.sub('', data)
    # 中文文本  去除特殊字符
    print(data)
    data = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', data)
    print(data)
    # 分词
    data = jieba.lcut(data, use_paddle=True)
    data = clean_stopwords(data)
    data = vocab.to_idx(data)


    data = torch.LongTensor(data).to(device)
    min_len = 5 #最短长度
    if len(data) < min_len:
        # 如果最长序列的长度小于最小长度，进行填充
        # inputs = [F.pad(seq, (0, min_len - len(seq))) for seq in inputs]
        data = F.pad(data, (0, min_len - len(data)))
    data = torch.unsqueeze(data, dim = 0)
    y_hat = model(data)

    # # 获取log_softmax前的数值
    # y_hat = torch.logsumexp(y_hat, dim=1)
    # print(y_hat)
    # 获取softmax的概率

    y_hat = torch.exp(y_hat)
    print(y_hat)
    print(y_hat.argmax(dim = 1))

if __name__=="__main__":
    str = "I want to fight, i am not happy"
    predict(str)

    # inputs = [torch.LongTensor(ex) for ex in data]
    # # 对batch中样本数据进行padding，使其长度相同
    # max_len = max([len(seq) for seq in inputs])
    # min_len = 5 #最短长度
    # if max_len < min_len:
    #     # 如果最长序列的长度小于最小长度，进行填充
    #     # inputs = [F.pad(seq, (0, min_len - len(seq))) for seq in inputs]
    #     inputs[0] = F.pad(inputs[0], (0, min_len - len(inputs[0])))

    # # 对序列进行填充
    # inputs = pad_sequence(inputs,batch_first=True)
    # inputs = inputs.to(device)
    # print(device)
    # y_hat = model(inputs)
    # y_hat = torch.exp(y_hat)
    # print(y_hat)