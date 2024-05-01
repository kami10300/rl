# 目前没有实现GPU的功能
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CustomDataset_cnn import CustomDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import keyboard
import numpy as np

kernelsizes = [(5, 50), (4, 50), (3, 50)]
lr = 1e-2
momentum = 1e-4
EPOCH = 20

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=kernel_size) for kernel_size in kernelsizes
        ])
        self.pooling = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        batch_size = x.size(0)
        conv_results = [conv(x) for conv in self.convs]
        # for conv in self.convs:
        #     print(conv(x).shape)
        x = torch.cat(conv_results, dim=2)  # 把卷积得到的结果进行拼接,利用torch.cat
        # print(x.shape)
        # keyboard.wait('enter')
        x = F.relu(self.pooling(x))
        # print(x.shape)
        # keyboard.wait('enter')
        x = x.view(batch_size, -1)   # 一定要把即将输入的向量扁平化，去掉两个维度
        x = torch.cat((x, torch.zeros(batch_size, 10).to(device)), dim = 1)   # 补全维度,使适配线性层的输入维度
        x = F.softmax(self.fc(x), dim = 1)
        # print(x)
        # keyboard.wait('enter')
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用{}训练".format(device))
model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)

def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    total_loss = 0
    total_correct = 0
    total_num = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target, target_item = data
        inputs, target, target_item = inputs.to(device), target.to(device), target_item.to(device)
        optimizer.zero_grad()

        # print(inputs.shape)
        # keyboard.wait('enter')

        # inputs的维度是 [32, 679, 50]
        # 如何train函数里面不修改的话，应该：
        # inputs = inputs.unsqueeze(1)
        # 在第二个维度上添加一个通道维度，将其变为 [32, 1, 679, 50]

        outputs = model(inputs)
        # print(outputs, target)
        loss = criterion(outputs, target)
        # keyboard.wait('enter')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim = 1)
        running_total += inputs.shape[0]
        total_num += inputs.shape[0]
        running_correct += (predicted == target_item).sum().item()
        total_correct += (predicted == target_item).sum().item()

        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_total = 0
            running_correct = 0
            running_loss = 0
    train_writer.add_scalar('Loss/train', total_loss / total_num, epoch)
    train_writer.add_scalar('Accuracy/train', total_correct / total_num, epoch)

def test(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    total_loss = 0
    total_correct = 0
    total_num = 0
    for data in val_loader:
        inputs, target, target_item = data
        inputs, target, target_item = inputs.to(device), target.to(device), target_item.to(device)
        # optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        # loss.backward()
        # optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim = 1)
        running_total += inputs.shape[0]
        total_num += inputs.shape[0]
        running_correct += (predicted == target_item).sum().item()
        total_correct += (predicted == target_item).sum().item()

    print('Test epoch: %d , loss: %.3f , acc: %.2f %% \n'
          % ((epoch + 1) / 2, running_loss / len(val_loader), 100 * running_correct / running_total))
    train_writer.add_scalar('Loss/test', total_loss / total_num, epoch)
    train_writer.add_scalar('Accuracy/test', total_correct / total_num, epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bz', type = int, default = 16)
    args = parser.parse_args()
    BATCH = args.bz
    print(BATCH)

    train_dataset = CustomDataset("Dataset//train.txt")
    val_dataset = CustomDataset("Dataset//validation.txt")
    train_loader = DataLoader(train_dataset, batch_size = BATCH, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH)

    TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
    train_log_dir = "logs/cnn/train/cnn_bz_" + str(BATCH) + "_" + TIMESTAMP
    test_log_dir = "logs/cnn/test/cnn_bz_" + str(BATCH) + "_" + TIMESTAMP
    train_writer = SummaryWriter(train_log_dir)
    test_writer = SummaryWriter(test_log_dir)

    for epoch in range(EPOCH):
        train(epoch)
        if epoch % 2 == 1:
            test(epoch)

    train_writer.close()
    test_writer.close()