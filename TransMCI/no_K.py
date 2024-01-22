import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from util.util import MyDataset, train_eval, test, load_data
from model.transformer import TransformerModel

# 获取数据的维度
num_people = 137  # 数据中的人数
num_features = 116  # 特征数量
feature_dim = 137  # 特征形状

X, y = load_data()

# 划分数据集
train_features, test_features, train_labels, test_labels = train_test_split(X, y,
                                                                            test_size=0.2, random_state=42)

batch_size = 16

train_dataset = MyDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MyDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

input_dim = num_features  # 输入维度等于特征数量
output_dim = 2
nhead = 2  # Transformer头的数量
num_encoder_layers = 8  # 编码器层的数量
num_decoder_layers = 8  # 解码器层的数量
hidden_dim = 256  # 前馈神经网络的中间维度
dropout = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(feature_dim, nhead, num_encoder_layers, num_decoder_layers, hidden_dim, dropout,
                         output_dim).to(device)

# model = BiLSTM(feature_dim, hidden_dim, 4, output_dim).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

num_epochs = 60  # 根据需要调整训练周期

train_eval(num_epochs, model, train_loader, test_loader, criterion, optimizer, device)

test(model, test_loader, criterion, device, True)
