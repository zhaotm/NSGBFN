import torch
import torch.nn as nn


# 定义双向LSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义双向LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # 定义全连接层
        self.fc = nn.Linear(2 * hidden_size, output_dim)  # 乘以2是因为是双向LSTM
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 初始化LSTM隐藏状态
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 从LSTM的输出中获取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        out = self.softmax(out)
        return out
