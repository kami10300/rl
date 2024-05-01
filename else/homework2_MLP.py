# 目前没有实现GPU的功能
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CustomDataset_MLP import CustomDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import keyboard
import numpy as np

lr = 1e-2
momentum = 1e-4
EPOCH = 20

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用{}训练".format(device))
model = MLP().to(device)
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
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

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
    train_log_dir = "logs/mlp/train/mlp_bz_" + str(BATCH) + "_" + TIMESTAMP
    test_log_dir = "logs/mlp/test/mlp_bz_" + str(BATCH) + "_" + TIMESTAMP
    train_writer = SummaryWriter(train_log_dir)
    test_writer = SummaryWriter(test_log_dir)

    for epoch in range(EPOCH):
        train(epoch)
        if epoch % 2 == 1:
            test(epoch)

    train_writer.close()
    test_writer.close()