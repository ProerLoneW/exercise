import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义一个复杂函数
def complex_function(x):
    return np.arctan(x) + np.log(x**2 + 1) + np.exp(x) - 8 * x**2 + np.sin(x)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个线性层
        self.relu = nn.ReLU()                          # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二个线性层
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 生成数据
def generate_data(n_samples, x_range=(-10, 10)):
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = complex_function(x)
    return x, y

# 训练模型
def train_model(model, x_train, y_train, epochs=2000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 转换为PyTorch张量
    x_tensor = torch.FloatTensor(x_train).view(-1, 1)
    y_tensor = torch.FloatTensor(y_train).view(-1, 1)
    
    losses = []
    
    for epoch in range(epochs):
        # 前向传播
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # 打印训练进度
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 可视化结果
def visualize_results(model, x_train, y_train, x_test, y_test):
    model.eval()
    with torch.no_grad():
        x_train_tensor = torch.FloatTensor(x_train).view(-1, 1)
        y_pred_train = model(x_train_tensor).numpy().flatten()
        
        x_test_tensor = torch.FloatTensor(x_test).view(-1, 1)
        y_pred_test = model(x_test_tensor).numpy().flatten()
    
    plt.figure(figsize=(12, 10))
    
    # 训练数据拟合结果
    plt.subplot(2, 1, 1)
    plt.scatter(x_train, y_train, color='blue', alpha=0.5, label='真实值')
    plt.scatter(x_train, y_pred_train, color='red', alpha=0.5, label='预测值')
    plt.title('训练集拟合结果')
    plt.legend()
    
    # 测试数据拟合结果
    plt.subplot(2, 1, 2)
    plt.scatter(x_test, y_test, color='blue', alpha=0.5, label='真实值')
    plt.scatter(x_test, y_pred_test, color='red', alpha=0.5, label='预测值')
    plt.title('测试集拟合结果')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成训练和测试数据
    x_train, y_train = generate_data(1000)
    x_test, y_test = generate_data(200)
    
    # 创建模型
    input_size = 1
    hidden_size = 4096  # 可以调整隐藏层大小以改变模型复杂度
    output_size = 1
    
    model = SimpleNN(input_size, hidden_size, output_size)
    
    # 训练模型
    losses = train_model(model, x_train, y_train, epochs=2000, lr=0.01)
    
    # 可视化结果
    visualize_results(model, x_train, y_train, x_test, y_test)
    
    # 计算测试集上的均方误差
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test).view(-1, 1)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        y_pred = model(x_test_tensor)
        test_loss = nn.MSELoss()(y_pred, y_test_tensor)
    
    print(f'测试集上的均方误差: {test_loss.item():.4f}')