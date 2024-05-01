import torch
import numpy as np
import keyboard
from torch.utils.data import Dataset
from read_word2vec import read_word2vec

word_vectors_dict = read_word2vec()

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding = 'utf8') as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].strip().split()
        inputs = np.array([np.zeros(50) if item not in word_vectors_dict else word_vectors_dict[item] for item in sample[1:]])
        index = inputs.shape[0]
        inputs = np.pad(inputs, ((0, 679 - index), (0, 0)), mode='constant', constant_values=0)
        inputs[678, 0] = index - 1

        # print(inputs.shape)
        # keyboard.wait('enter')

        # inputs = torch.tensor(inputs).unsqueeze(0)
        inputs = torch.tensor(inputs).float()
        # rnn中不需要在tensor中加上一个通道数的维度

        if sample[0] == '1':
            target = torch.tensor([0.0, 1.0])
        else:
            target = torch.tensor([1.0, 0.0])
        return inputs, target, int(sample[0])