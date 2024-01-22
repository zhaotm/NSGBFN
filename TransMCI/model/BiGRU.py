import torch
import torch.nn as nn


# 定义一个简单的RNN模型
class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RnnModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
