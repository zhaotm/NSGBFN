import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, roc_auc_score, \
    confusion_matrix, accuracy_score, roc_curve
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def get_features(self):
        return self.X


def load_data():
    # 从Excel文件中读取数据
    X = pd.read_excel('./data/data_1.xlsx').to_numpy()
    y = pd.read_excel('./data/label.xlsx').to_numpy()
    # 获取数据的维度
    num_people = 137  # 数据中的人数
    num_features = 137  # 特征数量
    feature_dim = 116  # 特征形状

    X = X.reshape(num_people, num_features, feature_dim).astype('float32')
    return X, y


def train_eval(num_epochs, model, train_loader, test_loader, criterion, optimizer, device):
    l1 = []
    l2 = []
    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1:02}/{num_epochs} | train loss: {train_loss:.4f}')
        test_loss = test(model, test_loader, criterion, device)
        l1.append(train_loss)
        l2.append(test_loss)
        if epoch % 10 == 0:
            print(f"test loss: {test_loss}")

    x = [i + 1 for i in range(len(l1))]
    plt.figure(figsize=(16, 8))
    plt.plot(x, l1, color='#082567')
    plt.plot(x, l2, color="#0dbf8c")
    plt.show()


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for _, (X, y) in enumerate(loader):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y.squeeze(dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)

    return running_loss / len(loader.dataset)


def test(model, loader, criterion, device, save=False):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_scores = []
    with torch.no_grad():
        for X, y in loader:
            y_true.extend(y.numpy())
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y.long().view(-1))
            running_loss += loss.item() * X.size(0)
            y_scores.append(y_pred.cpu().numpy())

    if save:
        y_pred_labels = np.argmax(np.concatenate(y_scores), axis=1)
        acc, sen, spe, ppv, npv = metric_eval(y_true, y_pred_labels)
        print(f'acc: {acc:.4f}\nsen: {sen:.4f}\nspe: {spe:.4f}\n'
              f'ppv: {ppv:.4f}\nnpv:{npv:.4f}')
        metric_df = pd.DataFrame([[acc, sen, spe, ppv, npv]], columns=['acc', 'sen', 'spe', 'ppv', 'npv'])
        metric_df = metric_df.round(4)
        with pd.ExcelWriter('./result/metric.xlsx', mode='a', engine='openpyxl') as writer:
            metric_df.to_excel(writer, index=False, header=False)
    return running_loss / len(loader.dataset)


def metric_eval(y_test, y_pred):
    # 计算分类器的准确率
    acc = accuracy_score(y_test, y_pred)

    # 计算分类器的召回率
    sensitivity = recall_score(y_test, y_pred)

    # 计算特异性
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # 计算正预测值
    ppv = precision_score(y_test, y_pred)

    # 计算负预测值
    npv = tn / (tn + fn)

    return acc, sensitivity, specificity, ppv, npv
