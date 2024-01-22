from torch import nn


class Cov1(nn.Module):

    def __init__(self, input_size, output_size):
        super(Cov1, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_size),
            nn.MaxPool1d(kernel_size=3),
        )

    def forward(self, x):
        return self.cov1(x)


class Cov2(nn.Module):

    def __init__(self, input_size, output_size):
        super(Cov2, self).__init__()
        self.cov2 = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cov2(x)


class Model(nn.Module):

    def __init__(self, input_size, output_size, dropout=0.0):
        super(Model, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_size, 256),
                                    # nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))

        self.layer2 = nn.Sequential(nn.Linear(256, 512),
                                    nn.ReLU())

        self.cov1 = Cov1(116, 64)

        self.cov2 = Cov2(64, 32)

        self.layer4 = nn.Sequential(nn.Linear(5440, 256),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))
        self.layer5 = nn.Sequential(nn.Linear(256, output_size),
                                    nn.Softmax(dim=1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.cov1(out)
        out = self.cov2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
