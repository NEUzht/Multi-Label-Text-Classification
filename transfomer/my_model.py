import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class My_model(nn.Module):
    def __init__(self, vocab_size=40000, embed_dim=128, dropout=0.7, fusion_size = 64, class_num=2, device = torch.device("cuda"),):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)
        #TextCNN
        kernel_num=3
        kernel_size=[2,3,4]
        ci = 1  # input chanel size
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], self.embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], self.embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], self.embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, fusion_size)
        #TextRNN
        self.hidden_size = 128
        self.layer_num = 3
        self.bidirectional = True
        self.device = device
        # if static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
        #     self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)
        self.lstm = nn.LSTM(self.embed_dim, # x的特征维度,即embedding_dim
                            self.hidden_size,# 隐藏层单元数
                            self.layer_num,# 层数
                            batch_first=True,# 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional) # 是否用双向
        self.fc2 = nn.Linear(self.hidden_size * 2, fusion_size) if self.bidirectional else nn.Linear(self.hidden_size, fusion_size)
      
        #Transformer
        num_heads = 4
        d_hid = 256
        nlayers = 2
        self.pos_encoder = PositionalEncoding(self.embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(self.embed_dim, num_heads, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(self.embed_dim, fusion_size)
        self.init_weight()
        
        self.fc = nn.Linear(3*fusion_size, class_num)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embedding(x)
        
        #RNN
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size).to(self.device) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size).to(self.device) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size).to(self.device)

        #CNN
        # x: (batch, sentence_length, embed_dim)
        x1 = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1_1 = self.conv_and_pool(x1, self.conv11)  # (batch, kernel_num)
        x1_2 = self.conv_and_pool(x1, self.conv12)  # (batch, kernel_num)
        x1_3 = self.conv_and_pool(x1, self.conv13)  # (batch, kernel_num)
        x1 = torch.cat((x1_1, x1_2, x1_3), 1)  # (batch, 3 * kernel_num)
        x1 = self.dropout(x1)
        x1 = self.fc1(x1)
        
        #RNN
        out, (hn, cn) = self.lstm(x, (h0, c0))
        x2 = self.fc2(out[:, -1, :])
        
        #transformer
        #[seq_len, batch_size]
        x3 = x.transpose(0, 1) * math.sqrt(self.embed_dim)
        x3 = self.pos_encoder(x3)
        x3 = self.transformer_encoder(x3)
        # print("***",output.shape,"***")
        x3 = self.decoder(x3)
        # print("***",output.shape,"***")
        x3 = torch.mean(x3, dim=0)
        # print(f'x1.shape{x1.shape} x2.shape{x2.shape} x3.shape{x3.shape}')
        #融合
        features = torch.cat((x1,x2,x3),dim=1)
        logit = F.log_softmax(self.fc(features), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":

    model = My_model(60000).cuda()
    x = torch.randint(0,60000,(4,22)).cuda()
    print(x.shape)
    y = model(x)
    print(y.shape)
    # print(y)
