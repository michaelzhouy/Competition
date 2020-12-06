# -*- coding: utf-8 -*-
# @Time     : 2020/12/6 18:16
# @Author   : Michael_Zhouy
import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.utils.data import Dataset, DataLoader


class tabularDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


train_ds = tabularDataset(X, Y)


class tabularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(14, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 2)
        self.bn_in = nn.BatchNorm1d(14)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x_in):
        x = self.bn_in(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x


# 训练前指定使用的设备
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
print(DEVICE)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 实例化模型
model = tabularModel().to(DEVICE)
print(model)

# 学习率
LEARNING_RATE = 0.01
# BS
batch_size = 1024
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# DataLoader加载数据
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

model.train()
# 训练10轮
TOTAL_EPOCHS = 100
# 记录损失函数
losses = []
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_dl):
        x = x.float().to(DEVICE)  # 输入必须未float类型
        y = y.long().to(DEVICE)  # 结果标签必须未long类型
        # 清零
        optimizer.zero_grad()
        outputs = model(x)
        # 计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
    print('Epoch : %d/%d,   Loss: %.4f' % (epoch+1, TOTAL_EPOCHS, np.mean(losses)))

model.eval()
correct = 0
total = 0
for i,(x, y) in enumerate(train_dl):
    x = x.float().to(DEVICE)
    y = y.long()
    outputs = model(x).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
print('准确率: %.4f %%' % (100 * correct / total))
