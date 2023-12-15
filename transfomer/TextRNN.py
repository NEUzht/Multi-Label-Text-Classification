import torch
import torch.nn as nn
import argparse
from torch.nn import functional as F

# 循环神经网络 (many-to-one)
class TextRNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size, layer_num, device, bidirectional = True, static = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.label_num = 8
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # if static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
        #     self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.lstm = nn.LSTM(self.embedding_dim, # x的特征维度,即embedding_dim
                            self.hidden_size,# 隐藏层单元数
                            self.layer_num,# 层数
                            batch_first=True,# 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional) # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2, self.label_num) if self.bidirectional else nn.Linear(self.hidden_size, self.label_num)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size).to(self.device) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size).to(self.device)

        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size).to(self.device) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size).to(self.device)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态?维度与h0和c0一样
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        out = self.fc(out[:, -1, :])
        out = F.log_softmax(out, dim=1)
        return out

parser = argparse.ArgumentParser(description='TextRNN text classifier')

parser.add_argument('-lr', type=float, default=0.01, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-vocab_size', type=int, default=100000)
parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
parser.add_argument('-hidden_size', type=int, default=64, help='lstm中神经单元数')
parser.add_argument('-layer-num', type=int, default=1, help='lstm stack的层数')
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-bidirectional', type=bool, default=False, help='是否使用双向lstm')
parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')

args = parser.parse_args()

if __name__ == "__main__":

    model = TextRNN(128, 100000, 64, 1).cuda()
    x = torch.randint(0,100000,(4,22)).cuda()
    print(x.shape)
    y = model(x)
    print(y.shape)
    print(y)