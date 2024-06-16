import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm
import os
import math

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  # 以字为键
            self.idx2word[self.idx] = word  # 以数值为键
            self.idx += 1

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()  # 继承类，初始化映射表
        self.file_list = []

    def get_file(self, filepath):
        for root, path, fil in os.walk(filepath):
            for txt_file in fil:
                self.file_list.append(root + txt_file)
        return self.file_list

    def get_data(self, batch_size):  # 读取文件，导入映射表
        # step 1
        tokens = 0
        for path in self.file_list:
            print(path)
            with open(path, 'r', encoding="ANSI") as f:
                for line in f.readlines():
                    # 把一些无意义的空格、段落符给去掉
                    line = line.replace(' ', '')
                    line = line.replace('\u3000', '')
                    line = line.replace('\t', '')
                    # jieba
                    words = jieba.lcut(line) + ['<eos>']
                    tokens += len(words)
                    for word in words:  # 构造彼此映射的关系
                        self.dictionary.add_word(word)
        # step 2
        ids = torch.LongTensor(tokens)  # 实例化一个LongTensor，命名为ids。遍历全部文本，根据映射表把单词转成索引，存入ids里
        token = 0
        for path in self.file_list:
            with open(path, 'r', encoding="ANSI") as f:
                for line in f.readlines():
                    line = line.replace(' ', '')
                    line = line.replace('\u3000', '')
                    line = line.replace('\t', '')
                    words = jieba.lcut(line) + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]  # 把每个词对应的索引存在ids里
                        token += 1
        # step 3 根据batchsize重构成一个矩阵
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids
    
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
class PositionalEncoding(nn.Module):  #定义位置编码和Transformer模型
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(embed_size, vocab_size)
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src, tgt):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        # tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src = self.encoder(src) * math.sqrt(self.embed_size)
        src = self.dropout_layer(src)
        src = self.pos_encoder(src)
        # tgt = self.encoder(tgt) * math.sqrt(self.embed_size)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt)
        output = self.decoder(output)
        return output




batch_size = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corpus = Corpus()  # 构造实例
corpus.get_file('./book1/')
ids = corpus.get_data(batch_size)  # 获得数据
vocab_size = len(corpus.dictionary)  # 词总数
src_vocab_size = len(corpus.dictionary)
trg_vocab_size = len(corpus.dictionary)
d_model = 512
nhead = 8
num_layers = 6# num_layers
num_decoder_layers = 6
# dim_feedforward = 1024# hidden_size
seq_length = 50# seq_length


# '''训练'''
# embed_size = 256#增加每个词涵盖的特征数，提高结果精准度
hidden_size = 512#增加神经元数量
# num_layers = 3#增加隐藏层
num_epochs = 1#增加训练次数
# batch_size = 50
# seq_length = 30  # 序列长度，我认为是与前多少个词具有相关程度
learning_rate = 0.001
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# corpus = Corpus()  # 构造实例
# corpus.get_file('./ch1/')
# ids = corpus.get_data(batch_size)  # 获得数据
# vocab_size = len(corpus.dictionary)  # 词总数




whether_train = 1

if whether_train:
    model = TransformerModel(vocab_size,embed_size=256,num_heads=8,num_layers=6,hidden_dim=512,dropout=0.1).to(device)

    cost = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))  # 参数矩阵初始化(h,c)

        for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):  # 打印循环中的进度条
            inputs = ids[:, i:i + seq_length].to(device)  # 训练集的输入
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)  # 训练集的结果

            states = [state.detach() for state in states]
            # detach返回一个新的tensor，相当于可以切断反向传播的计算
            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))

            model.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            clip_grad_norm_(model.parameters(), 0.5)  # 避免梯度爆炸
            optimizer.step()
    '''保存模型'''
    save_path = './model_path/model_TF.pt'
    torch.save(model, save_path)
else:
    model = torch.load('./model_path/model_TF.pt')

'''生成文本'''
num_samples = 500  # 生成文本的长度，可以认为是包含单词的个数
article = str()  # 输出文本的容器
'''自定义输入'''
input_para = '青光闪动，一柄青钢剑倏地刺出，指向在年汉子左肩'
input_words = jieba.lcut(input_para)
print(input_words)
input_len = len(input_words)
input_lst = []
for input_word in input_words:
    lst = [corpus.dictionary.word2idx[input_word]]
    input_lst.append(lst)
_input = torch.Tensor(input_lst).to(device).to(dtype=torch.long)
state = (torch.zeros(num_layers, input_len, hidden_size).to(device),
         torch.zeros(num_layers, input_len, hidden_size).to(device))  # 初始化参数
prob = torch.ones(vocab_size)  # 对应模型中的outputs，相当于单词的概率分布
article = ''.join(input_para)
for i in range(num_samples):
    output, state = model(_input, state)
    prob = output.exp()
    word_id = torch.multinomial(prob, num_samples=1)
    for j in word_id:
        word_value = j.item()
    word_tensor = torch.Tensor([word_value]).to(device).to(dtype=torch.long)
    _input_squeeze = _input.squeeze()
    _input = _input_squeeze[1:]
    _input = torch.cat((_input, word_tensor), 0).unsqueeze(1).to(dtype=torch.long)
    word = corpus.dictionary.idx2word[word_value]
    word = '\n' if word == '<eos>' else word
    article += word
print(article)
# '''文本保存'''
# txt_name = './文本生成/'+str(num_samples)+'.txt'
# with open(txt_name, 'w', encoding="utf-8") as gen_file:
#     gen_file.write(article)