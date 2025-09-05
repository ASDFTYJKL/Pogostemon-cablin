import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    dataset = dataset[permutation, :]
    labels = labels[permutation]
    return dataset, labels


# 数据加载和处理

#示例数据，仅仅证明模型可以正常运行，用户可以导入自己需要的数据

data_dir = "./batch_1"
label_0_path = os.path.join(data_dir, 'label_0_examples.npy')
label_1_path = os.path.join(data_dir, 'label_1_examples.npy')
label_2_path = os.path.join(data_dir, 'label_2_examples.npy')
label_3_path = os.path.join(data_dir, 'label_3_examples.npy')

# 加载数据
label_0_pixels = np.load(label_0_path)
label_1_pixels = np.load(label_1_path)
label_2_pixels = np.load(label_2_path)
label_3_pixels = np.load(label_3_path)

# 随机抽取样本

#示例样本，仅仅证明模型可以正常运行

np.random.seed(42)
label_0_test_indices = np.random.choice(label_0_pixels.shape[0], 100, replace=False)
label_1_test_indices = np.random.choice(label_1_pixels.shape[0], 100, replace=False)
label_2_test_indices = np.random.choice(label_2_pixels.shape[0], 100, replace=False)
label_3_test_indices = np.random.choice(label_3_pixels.shape[0], 300, replace=False)

label_0_test = label_0_pixels[label_0_test_indices]
label_1_test = label_1_pixels[label_1_test_indices]
label_2_test = label_2_pixels[label_2_test_indices]
label_3_test = label_3_pixels[label_3_test_indices]

negative_test_samples = np.vstack((label_0_test, label_1_test, label_2_test))
positive_test_samples = label_3_test

X_test = np.vstack((positive_test_samples, negative_test_samples))
y_test = np.hstack((np.ones(positive_test_samples.shape[0]), np.zeros(negative_test_samples.shape[0])))

# 剩余数据
remaining_label_0 = np.delete(label_0_pixels, label_0_test_indices, axis=0)
remaining_label_1 = np.delete(label_1_pixels, label_1_test_indices, axis=0)
remaining_label_2 = np.delete(label_2_pixels, label_2_test_indices, axis=0)
remaining_label_3 = np.delete(label_3_pixels, label_3_test_indices, axis=0)

# 划分训练集和验证集
label_0_sample_indices = np.random.choice(remaining_label_0.shape[0], 200, replace=False)
label_1_sample_indices = np.random.choice(remaining_label_1.shape[0], 200, replace=False)
label_2_sample_indices = np.random.choice(remaining_label_2.shape[0], 200, replace=False)
label_3_sample_indices = np.random.choice(remaining_label_3.shape[0], 600, replace=False)

label_0_sample = remaining_label_0[label_0_sample_indices]
label_1_sample = remaining_label_1[label_1_sample_indices]
label_2_sample = remaining_label_2[label_2_sample_indices]
label_3_sample = remaining_label_3[label_3_sample_indices]

negative_samples = np.vstack((label_0_sample, label_2_sample, label_1_sample))
positive_samples = label_3_sample

positive_sample_count = positive_samples.shape[0]
negative_sample_count = negative_samples.shape[0]
upsampled_positive_samples = np.vstack(
    [positive_samples[np.random.choice(positive_sample_count, size=negative_sample_count, replace=True)] for _ in
     range((negative_sample_count // positive_sample_count) + 1)])
upsampled_positive_samples = upsampled_positive_samples[:negative_sample_count]

X = np.vstack((upsampled_positive_samples, negative_samples))
y = np.hstack((np.ones(upsampled_positive_samples.shape[0]), np.zeros(negative_samples.shape[0])))

# 打乱数据
shuffle_indices = np.random.permutation(np.arange(len(y)))
X = X[shuffle_indices]
y = y[shuffle_indices]

# 标准化数据
for i in range(X.shape[0]):
    X[i, :] = X[i, :] / np.max(X[i, :])

X = X.reshape(X.shape[0], -1)
X = X.reshape(-1, 1, X.shape[1])

for i in range(X_test.shape[0]):
    X_test[i, :] = X_test[i, :] / np.max(X_test[i, :])

X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test.reshape(-1, 1, X_test.shape[1])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义一维卷积神经网络
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer7 = nn.Sequential(
            nn.Conv1d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048 * 2, 256)  # 根据最终特征图大小更新输入
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(x.size(0), -1)  # 展平成全连接层输入
        x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建网络实例并移动到 GPU
net = CNN1D().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 记录训练过程中的损失和准确率
train_losses = []
val_losses = []
val_accuracies = []
train_accuracies = []

best_val_loss = float('inf')
best_model_path = 'example_model_zhaoqing_training.pth'
prev_val_loss = float('inf')
stable_threshold = 0.01  # 验证损失变化阈值

# 训练模型
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted_train = outputs.round()
        total_train += labels.size(0)
        correct_train += (predicted_train.view(-1) == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    net.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            val_loss += loss.item()
            predicted = outputs.round()
            total_val += labels.size(0)
            correct_val += (predicted.view(-1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val

    # 判断验证损失是否下降平稳
    val_loss_change = abs(prev_val_loss - val_loss)
    if val_loss_change < stable_threshold:
        torch.save(net.state_dict(), best_model_path)
        print(f"Validation loss has stabilized. Saved model at Epoch {epoch + 1}.")

    prev_val_loss = val_loss

    # 每五轮记录一次数据点
    if (epoch + 1) % 5 == 0 or epoch == 0:
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss:{val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy:{val_accuracy:.4f}")

# 绘制训练和验证损失与准确率
epochs_to_plot = list(range(0, num_epochs, 5))
if (num_epochs - 1) % 5 != 0:
    epochs_to_plot.append(num_epochs - 1)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax2 = ax1.twinx()

ax1.plot(epochs_to_plot, train_losses, 'g-', label='Train Loss')
ax1.plot(epochs_to_plot, val_losses, 'b-', label='Val Loss')
ax2.plot(epochs_to_plot, train_accuracies, 'r--', label='Train Accuracy')
ax2.plot(epochs_to_plot, val_accuracies, 'k--', label='Val Accuracy')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Training and Validation Loss and Accuracy')
plt.show()
net.load_state_dict(torch.load(best_model_path))
net.eval()

# 测试集评估
test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        test_loss += loss.item()
        predicted = outputs.round()
        total_test += labels.size(0)
        correct_test += (predicted.view(-1) == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = correct_test / total_test

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")