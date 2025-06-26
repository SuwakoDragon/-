import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
data = pd.read_csv('airline-passengers.csv')
data = data.rename(columns={'Month': 'ds', 'Passengers': 'y'})
data['ds'] = pd.to_datetime(data['ds'])

# 初始数据可视化
plt.figure(figsize=(10, 6))
plt.plot(data['ds'], data['y'])
plt.title('Airline Passengers Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

# 压缩数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['y']])


# 训练集、测试集创建
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
test_size = int(0.25 * len(X))
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# GRU模型构建 训练数据集
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, 2*hidden_size, batch_first=True)
        self.gru3 = nn.GRU(2*hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        out3, _ = self.gru3(out2)
        out = self.fc(out3[:, -1, :])
        return out

# 设备设置 训练设置
device = 'cuda:0'
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
model = GRUModel(input_size=1, hidden_size=50, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 20000
loss_list = []
with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pbar.set_postfix(Epoch=epoch + 1, Loss=loss.item())
        pbar.update(1)

# 保存与加载
torch.save(model.state_dict(), 'GRU.pth')
print('training result is saved')

loaded_model = GRUModel(input_size=1, hidden_size=50, output_size=1).to(device)
loaded_model.load_state_dict(torch.load('GRU.pth'))
loaded_model.eval()
print("loaded")
model.eval()

# 预测与可视化
predicted = model(X_test)
predicted_stock_price = scaler.inverse_transform(predicted.cpu().detach().numpy())
y_test_actual = scaler.inverse_transform(y_test.cpu().reshape(-1, 1).detach().numpy())
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, color='blue', label='true')
plt.plot(predicted_stock_price, color='red', label='predict')
plt.title('Airline Passengers Forecast')
plt.xlabel('time')
plt.ylabel('people numbers')
plt.legend()
plt.show()


# loss函数值变化可视化
plt.figure(figsize=(10, 5))
plt.plot(loss_list, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()