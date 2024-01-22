import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from util.util import MyDataset, train_eval, test, load_data
from model.transformer import TransformerModel
from model.BiGRU import RnnModel

# 获取数据的维度
num_people = 137  # 数据中的人数
num_features = 137  # 特征数量
feature_dim = 116  # 特征形状

num_folds = 5

# Create a StratifiedKFold object
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=126357)

batch_size = 16
input_dim = feature_dim  # 输入维度等于特征数量
output_dim = 2  # 类别数
nhead = 1  # Transformer头的数量
num_encoder_layers = 2  # 编码器层的数量
num_decoder_layers = 2  # 解码器层的数量
hidden_dim = 256  # 前馈神经网络的中间维度
dropout = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.00001
num_epochs = 40


X, y = load_data()

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    train_features, test_features = X[train_index], X[test_index]
    train_labels, test_labels = y[train_index], y[test_index]

    train_dataset = MyDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = RnnModel(input_dim, hidden_dim, 2, output_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print(f"Fold {fold + 1} / {num_folds}")

    train_eval(num_epochs, model, train_loader, test_loader, criterion, optimizer, device)

    test(model, test_loader, criterion, device, True)

