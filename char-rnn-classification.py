import glob
import unicodedata
import torch
import string
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%matplotlib inline




all_filenames = glob.glob('../data/names/*.txt')
#print(all_filenames)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
#print(n_letters)
#print(all_letters)
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

#print(unicode_to_ascii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}#just the dictionary for the output
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in all_filenames:
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
#print('n_categories =', n_categories)
#print(all_categories)



# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    #print(n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor
# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor




# 没有定义loss函数？
#定义模型
class RNN(nn.Module):  # 定义rnn模型
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()  # 用于定义

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # 定义的生成隐藏层矩阵的维度
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # 定义生成输出的矩阵的维度
        self.softmax = nn.LogSoftmax(dim=1)  # logsoftmax的定义 记得加上dim=1，因为之后的版本softmax不在有默认参数了

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)  # 将隐藏层和输入层并入到一个矩阵中，同时使用W_h矩阵
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)#dim=1:[tensor(0.12),tensor(0.23)...]
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128#隐藏层的维度
#n_letters=57,n_hidden=128,n_categories=18
rnn = RNN(n_letters, n_hidden, n_categories)#n_categories作为输出，是要判断对应句子是哪一种语言

'''
input = Variable(letter_to_tensor('A'))
hidden = rnn.init_hidden()

output, next_hidden = rnn(input, hidden)
print('output.size =', output.size())
print(output)
print(next_hidden)
'''
'''
input = line_to_tensor('Albert')    
#print(input)
print(Variable(input))
print(input.shape)

'''
#定义从output产出结果的函数
def categoty_from_output(output):
    top_n, top_i = output.data.topk(1)#最大的那一个
    category_i = top_i[0][0]
    #print('top_n:', top_n)
    #print('top_i:', top_i)
    return all_categories[category_i], category_i

#print(categoty_from_output(output))
'''
'''
#定义随机训练的方式
def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = line_to_tensor((line))
    #print('category_tensor:', category_tensor) #category_tensor: tensor([5])
    #print('line_tensor:', line_tensor)
    return category, line, category_tensor, line_tensor
'''
for i in range(10):
    category, line, category_tensor, line_tensor = random_training_pair()
    print("category=", category, '/line =', line)
'''

#训练网络
criterion = nn.NLLLoss()#定义loss函数 输入是一个对数概率向量，和一个目标标签

learning_rate = 0.005#定义学习率
#print(rnn.parameters()) 应该是对应的rnn模型的参数 即n_letters, n_hiddens, n_catrgories
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)#定义优化器

'''
Each loop of training will:#每次大训练都有下列7个步骤

    Create input and target tensors#制造输入和目标向量
    Create a zeroed initial hidden state#制造一个全0的初始化的隐藏层
    Read each letter in and#一次读取每一个字母，并保留这次的隐藏层以便下一次的使用
        Keep hidden state for next letter
    Compare final output to target#比较此次的预测值和真实值的区别
    Back-propagate#执行反向传播算法
    Return the output and loss#返回最终结果和loss值


'''

#category_tensor对应的是此次语言的索引，即真实值，line_tensor对应的是此次的总input，每一次的小input是一个line里的每个字母对应的tensor
def train(category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)#need to learning
    loss.backward()

    optimizer.step()

    return output, loss.data#loss.data itself is the dim==1 tensor


'''
正式训练模型
'''

n_epochs = 100000
print_every = 5000
plot_every = 1000

#保证追踪每一次画图的loss值
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return ('%dm %d' %(m, s))

start =time.time()

right=0

for epoch in range(1, n_epochs+1):
    category, line, category_tensor, line_tensor = random_training_pair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    guess, _ = categoty_from_output(output)
    if(guess==category):
        right+=1
    if(epoch % print_every == 0):
        guess, guess_i = categoty_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line,  guess, correct))
        print('correct_rate:', 1.0*right/print_every)
        right = 0

    if(epoch%plot_every==0):
        all_losses.append(current_loss / plot_every)
        current_loss = 0

print(all_losses)
#最后的正确率是61.56%

plt.figure()
plt.plot(all_losses)

#了解如何查看中间的W_h和W_o
#if there is anybody know how to print the result（like W_h and W_o） or save them in the cjeckpoint and use them next time
#hoping you can tell me ☻
