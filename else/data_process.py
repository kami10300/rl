import torch

# 创建大小为 (32, 3, 1) 的输入数据
input_data = torch.randn(32, 3, 1)

# 使用 torch.view() 函数重新调整形状
input_data_reshaped = input_data.view(32, -1)

# 或者使用 torch.reshape() 函数
# input_data_reshaped = torch.reshape(input_data, (32, -1))

# 查看调整后的输入数据的形状
print("调整后的输入数据形状:", input_data_reshaped.shape)
