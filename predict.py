import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

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
# 加载预训练模型
device = torch.device("cpu")
model = CNN1D().to(device)
model.load_state_dict(torch.load('model_zhaoqing_finetune.pth', map_location=device))
model.eval()

# 数据路径
npy_dir = r"./"
label_dir =r"./"

# 获取npy文件列表
npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

# 预测并保存结果
for npy_file in npy_files:
    # 加载数据
    npy_path = os.path.join(npy_dir, npy_file)
    data1 = np.load(npy_path)
    data = data1.reshape(data1.shape[0]* data1.shape[1], 1, 256)
    # 数据预处理
    for i in range(data.shape[0]):
        data[i, :] = data[i, :] / np.max(data[i, :])

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    print(data.shape)
    print(data_tensor.shape)

    # 预测
    with torch.no_grad():
        output = model(data_tensor)
        prediction = output.cpu().numpy().round().astype(int)

    # 变形预测结果
    prediction = prediction.reshape(data1.shape[0], data1.shape[1])

    post_processed_prediction = np.zeros_like(prediction)
    for i in range(1, prediction.shape[0] - 1):
        for j in range(1, prediction.shape[1] - 1):
            neighborhood = prediction[i-1:i+2, j-1:j+2].flatten()
            counts = np.bincount(neighborhood)
            most_common = np.argmax(counts)
            if counts[most_common] >= 5:
                post_processed_prediction[i, j] = most_common
            else:
                post_processed_prediction[i, j] = prediction[i, j]

    # 获取对应的标签图片路径
    label_image_path = os.path.join(label_dir, npy_file.replace('.npy', '.png'))

    # 保存预测结果为图片
    prediction_image = Image.fromarray(post_processed_prediction * 255).convert('L')  # 转为灰度图
    prediction_image.save(label_image_path.replace('.png', '_prediction.png'))

    print(f"Prediction saved for {npy_file} as {label_image_path.replace('.png', '_prediction.png')}")