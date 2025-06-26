import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F  # 新增导入

# 数据加载和预处理
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

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['y']])


# 准备数据集
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

# 转换为PyTorch张量
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# =============== TCN模块定义 ===============
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        # 因果卷积：只使用左侧的时间点进行卷积（保证时序因果性）
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # 防止过拟合

    def forward(self, x):
        # 进行因果卷积
        out = self.conv(x)
        # 截断右侧填充部分以保持序列长度不变
        out = out[:, :, :-self.conv.padding[0]] if self.conv.padding[0] != 0 else out
        out = self.relu(out)
        return self.dropout(out)


# =============== TCN-GRU混合模型定义 ===============
class TCN_GRU_Model(nn.Module):
    def __init__(self, input_size=1, tcn_channels=64, gru_hidden_size=50, output_size=1):
        super(TCN_GRU_Model, self).__init__()

        # =============== TCN模块部分 ===============
        self.tcn_block1 = TCNBlock(input_size, tcn_channels, kernel_size=5, dilation=1)
        self.tcn_block2 = TCNBlock(tcn_channels, tcn_channels // 2, kernel_size=3, dilation=2)
        self.tcn_block3 = TCNBlock(tcn_channels // 2, tcn_channels // 4, kernel_size=3, dilation=4)

        # =============== GRU模块部分 ===============
        # 保持与原始代码类似的多层GRU结构
        self.gru1 = nn.GRU(tcn_channels // 4, gru_hidden_size, batch_first=True)
        self.gru2 = nn.GRU(gru_hidden_size, 2 * gru_hidden_size, batch_first=True)
        self.gru3 = nn.GRU(2 * gru_hidden_size, gru_hidden_size, batch_first=True)

        # 输出层
        self.fc = nn.Linear(gru_hidden_size, output_size)

    def forward(self, x):
        # 调整维度: [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        # =============== TCN处理流程 ===============
        tcn_out = self.tcn_block1(x)
        tcn_out = self.tcn_block2(tcn_out)
        tcn_out = self.tcn_block3(tcn_out)

        # 恢复维度: [batch, channels, seq_len] -> [batch, seq_len, channels]
        tcn_out = tcn_out.permute(0, 2, 1)

        # =============== GRU处理流程 ===============
        out1, _ = self.gru1(tcn_out)
        out2, _ = self.gru2(out1)
        out3, _ = self.gru3(out2)

        # =============== 输出预测 ===============
        out = self.fc(out3[:, -1, :])
        return out


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 转移数据到设备
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# 初始化模型
model = TCN_GRU_Model(
    input_size=1,
    tcn_channels=64,
    gru_hidden_size=64,
    output_size=1
).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# =============== 添加学习率调度器 ===============
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,
    patience=100,
    min_lr=1e-5,
    verbose=True
)

# 训练
epochs = 3000
loss_list = []
best_loss = float('inf')

# 添加训练进度条
with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train)

        # 计算损失
        loss = criterion(outputs.squeeze(), y_train)

        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 更新学习率
        scheduler.step(loss)

        # 记录损失
        loss_list.append(loss.item())

        # 更新进度条
        pbar.set_postfix(Epoch=epoch + 1, Loss=f"{loss.item():.6f}")
        pbar.update(1)

        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_tcn_gru_model.pth')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_list, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# 加载最佳模型
model.load_state_dict(torch.load('best_tcn_gru_model.pth'))
model.eval()

# 预测
with torch.no_grad():
    train_predict = model(X_train)
    test_predict = model(X_test)

    # 反归一化
    train_predict = scaler.inverse_transform(train_predict.cpu().numpy())
    y_train_actual = scaler.inverse_transform(y_train.cpu().reshape(-1, 1).numpy())
    test_predict = scaler.inverse_transform(test_predict.cpu().numpy())
    y_test_actual = scaler.inverse_transform(y_test.cpu().reshape(-1, 1).numpy())

# 可视化训练集拟合效果
plt.figure(figsize=(12, 6))
plt.plot(y_train_actual, color='blue', label='True Train')
plt.plot(train_predict, color='red', alpha=0.7, label='Predicted Train')
plt.title('Training Data: True vs Predicted')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# 可视化测试集预测效果
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, color='blue', label='True Test')
plt.plot(test_predict, color='red', alpha=0.7, label='Predicted Test')
plt.title('Test Data: True vs Predicted')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# 生成完整数据集预测
full_data = scaled_data.copy()
full_data = torch.tensor(full_data, dtype=torch.float32).view(1, -1, 1).to(device)
model.eval()
with torch.no_grad():
    # 为了预测整个序列，我们逐步预测
    predictions = []

    # 初始输入序列（使用第一组时间步）
    input_seq = full_data[0, :60, :].unsqueeze(0)

    for i in range(len(full_data[0]) - 60):
        pred = model(input_seq)
        predictions.append(pred.item())

        # 更新输入序列：移除第一个时间步，添加预测值作为新的最后一个时间步
        input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)), dim=1)

# 反归一化
predictions = np.array(predictions).reshape(-1, 1)
full_predictions = scaler.inverse_transform(predictions)

# 创建完整预测的时间索引
prediction_dates = data['ds'][60:].reset_index(drop=True)

# 可视化完整预测
plt.figure(figsize=(14, 7))
plt.plot(data['ds'], data['y'], color='blue', label='True Data')
plt.plot(prediction_dates, full_predictions, color='red', alpha=0.7, label='Predicted')
plt.title('Full Data: True vs Predicted')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()