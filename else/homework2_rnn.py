# 没有实现cuda,不过那个很简单
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CustomDataset_rnn import CustomDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import keyboard
import numpy as np

input_size = 50
hidden_size = 32
output_size = 2
num_layers = 1
lr = 1e-2
EPOCH = 20

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) #  batch维度放在第一位
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        # 初始化RNN隐藏状态和LSTM的日志状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        index = x[:, 678, 0]
        # print(index, int(index[0]), out[0, int(index[0]), :])
        # print(out.shape)
        # try:
        #     out = torch.stack([out[k, int(index[k]), :] for k in range(BATCH)], dim=0)
        # except IndexError:
        #     print(index, index.shape)
        out = torch.stack([out[k, int(index[k]), :] for k in range(BATCH)], dim=0)
        # print(out.shape, out)
        # keyboard.wait('enter')
        # print(out, out.shape)
        # keyboard.wait('enter')
        out = self.fc(out)   #  取最后一个时间步的输出作为模型的输出（合理的）
        # print(out)
        # keyboard.wait('enter')
        return F.softmax(out, dim=1)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用{}训练".format(device))
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    total_loss = 0
    total_correct = 0
    total_num = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target, target_item= data
        inputs, target, target_item= inputs.to(device), target.to(device), target_item.to(device)
        optimizer.zero_grad()

        # print(inputs)
        # keyboard.wait('enter')
        outputs = model(inputs)
        # print(outputs, target)
        # keyboard.wait('enter')
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

        # print(batch_idx)
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
    train_log_dir = "logs/rnn/train/rnn_bz_" + str(BATCH) + "_" + TIMESTAMP
    test_log_dir = "logs/rnn/test/rnn_bz_" + str(BATCH) + "_" + TIMESTAMP
    train_writer = SummaryWriter(train_log_dir)
    test_writer = SummaryWriter(test_log_dir)

    for epoch in range(EPOCH):
        train(epoch)
        if epoch % 2 == 1:
            test(epoch)

    train_writer.close()
    test_writer.close()