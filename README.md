# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence ,pad_sequence ,pack_sequence
inputs = [("LIHUA went to The TsinghUA University".split(),["N", "V", "V", "N","N","N"]),
          ("Liping went to technical school ".split(),["N","V","V","N","N"]),
          ("I work in the mall ".split(),["N","V","V","N","N"]),
          ("we both have bright future".split(),["N", "N", "V", "ADJ", "N"])]
START_TAG,STOP_TAG="START","STOP"
tag_to_ix = {"N": 0, "V": 1, "ADJ":2 ,START_TAG: 3, STOP_TAG: 4}
tag_size=len(tag_to_ix)


character_to_idx={}
word_to_idx={}
vocab=len(word_to_idx)
c_size=len(character_to_idx)
words_characters={}
for sen,tags in inputs:
    for word in sen:
        if word not in word_to_idx:
            words_characters[len(word_to_idx)] = list(word)
            word_to_idx[word]=len(word_to_idx)
            for c in word:
                character_to_idx[c]=len(character_to_idx)

def prepare_lookupTable(sequence,to_idx):
    idxs=[to_idx[word] for word in sequence]
    return torch.tensor(idxs)

# print(word_to_idx)
# for input in inputs:
#     sentence, tag = input[0], input[1]
#     print("{}对应的张量是{}".format(sentence,prepare_lookupTable(sentence,word_to_idx)))
#     print("{}对应的张量是{}".format(tag,prepare_lookupTable(tag,tag_to_ix)))

WORD_EMBEDDING_DIM = 6
CHARACTER_EMBEDDING_DIM = 4
HIDDEN_DIM = 6
CHARACTER_HIDDEN_DIM = 6
class LSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim,c_embedding_dim, hidden_dim,c_hidden_dim, vocab_size, c_size,tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.character_dim=c_embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.character_embeddings=nn.Embedding(c_size,c_embedding_dim)
        self.lstm_c = nn.LSTM(c_embedding_dim, c_hidden_dim) # LSTM_c 以每个字符的c_embeddings作为输入, 输出即为该单词对应字符级别的特征 输出维度为 c_hidden_dim 的隐藏状态值
        self.tag_lstm = nn.LSTM(word_embedding_dim + c_hidden_dim, hidden_dim) ## tag_lstm以word_embeddings和该词字符级别特征的拼接向量作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size) # 线性层将隐藏状态空间映射到标注空间
        self.hidden_tag = self.init_hidden(self.hidden_dim)
        self.hidden_character = self.init_hidden(c_hidden_dim)
    def init_hidden(self,hiddem_dim):
        # 一开始并没有隐藏状态所以我们要先初始化一个, 各个维度的含义是 (num_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence,word_characters):
        cat_embeds=[]
        for word in sentence:
            print("semtence是{}".format(sentence))
            print("word是{}".format(word))
            word_embed= self.word_embeddings(word)
            print("word_embed是{}".format(word_embed))
            word_character=word_characters[word.item()] #item() 方法是用来将只有一个元素的numpy数组或tensor张量转化为标量的方法
            print("word_c是{}".format(word_character))
            word_character_in = prepare_lookupTable(word_character, character_to_idx)
            print("word_c_in是{}".format(word_character_in))
            c_embeds = self.character_embeddings(word_character_in)
            print("c_embeds是{}".format(c_embeds))
            lstm_out_c, self.hidden_character = self.lstm_c(c_embeds.view(len(word_character_in), 1, -1), self.hidden_character)
            cat_emb=torch.cat((word_embed,self.hidden_character[0].view(-1)))
            cat_embeds.append(cat_emb)
        embeds = torch.cat(cat_embeds).view(len(sentence), 1, -1)
        lstm_out, self.hidden_tag = self.tag_lstm(embeds, self.hidden_tag)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        print("tag_space{}".format(tag_space))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(WORD_EMBEDDING_DIM, CHARACTER_EMBEDDING_DIM, HIDDEN_DIM, CHARACTER_HIDDEN_DIM, vocab,c_size,tag_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


with torch.no_grad():
    inputs1 = prepare_lookupTable(inputs[0][0], word_to_idx)
    tag_scores = model(inputs1, words_characters)
    print("训练前的分数：\n")
    print(tag_scores)
for epoch in range(2):  #  此例中我是随便设了一个值
    for input in inputs:
        # 第一步: 请记住Pytorch会累加梯度. 我们需要在训练每个实例前清空梯度
        sentence,tag=input[0],input[1]
        model.zero_grad()
        # 此外还需要清空 LSTM 的隐状态, # 将其从上个实例的历史中分离出来.
        model.hidden_tag = model.init_hidden(HIDDEN_DIM)
        model.hidden_character = model.init_hidden(CHARACTER_HIDDEN_DIM)
        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        sentence_in = prepare_lookupTable(sentence, word_to_idx)
        targets = prepare_lookupTable(tag, tag_to_ix)
        # print(sentence_in,targets)
        # 第三步: 前向传播.
        tag_scores = model(sentence_in, words_characters)
        # print(tag_scores)
        # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
with torch.no_grad():
    inputs1 = prepare_lookupTable(inputs[0][0], word_to_idx)
    tag_scores = model(inputs1, words_characters)
    print("训练前的分数：\n")
    print(tag_scores)
