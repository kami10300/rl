# dqn 经验回放 目标网络（target）
import random
import torch
import torch.nn as nn
import numpy as np
import gym

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,2)
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        return self.fc(inputs)

env = gym.envs.make('CartPole-v1')
env = env.unwrapped
net = MyNet()   # dqn
net2 = MyNet()  # target network

store_count = 0
store_size = 2000   # buffer size
decline = 0.6
learn_time = 0
update_time = 20
gama = 0.9
b_size = 1000  # batch size
store = np.zeros((store_size,10))   # 初始化buffer列中储存的s,a,s_,r
start_study = False

for _ in range(5000):
    s = env.reset()
    while True:
        if random.randint(0,100) < 100 * (decline ** learn_time):
            a = random.randint(0,1)
        else:
            out = net(torch.Tensor(s)).detach()
            a = torch.argmax(out).data.item()    # .data能够获取torch张量的底层数据，而取消梯度追踪
        _s, r, done, info = env.step(a)
        r = (env.theta_threshold_radians - abs(_s[2])) / env.theta_threshold_radians * 7  \
            + (env.x_threshold - abs(_s[0])) / env.x_threshold *0.3
        # 经验回放
        store[store_count % store_size][0:4] = s
        store[store_count % store_size][4:5] = a
        store[store_count % store_size][5:9] = _s
        store[store_count % store_size][9:10] = r
        store_count += 1
        s = _s

        if store_count > store_size:
            if learn_time % update_time == 0:
                net2.load_state_dict(net.state_dict())
            
            index = random.randint(0, store_size - b_size - 1)
            b_s = torch.Tensor(store[index : index + b_size, 0 : 4])
            b_a = torch.Tensor(store[index : index + b_size, 4 : 5]).long()
            b_s_ = torch.Tensor(store[index : index + b_size, 5 : 9])
            b_r = torch.Tensor(store[index : index + b_size, 9 : 10])

            q = net(b_s).gather(1, b_a)    # gather作用是按维度选择元素
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1)
            # 多用detach(),少用data()
            tq = b_r + gama * q_next
            loss = net.mls(q, tq)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()

            learn_time += 1
            if not start_study:
                print("start study!")
                start_study = True
                break
        if done:
            break

        env.render()