import argparse
from collections import namedtuple
from itertools import count
import matplotlib as plt
import os, sys, random
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import time
import LunarLanderContinuousv2
import highway_env
from torch.autograd import Variable
from torch import FloatTensor, LongTensor, ByteTensor
from lstm import LstmParam, LstmNetwork
'''
0:左 lane_left
1:直行 idele
2:右 lane_right
3:加速 faster
4：减速 slower
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument("--env_name",  default="ramp-v0")
parser.add_argument('--tau', default=0.05, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_episode', default=50, type=int)
parser.add_argument('--epoch', default=10, type=int)  # buffer采样的数据训练几次
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=500, type=int)  # replay buffer size
parser.add_argument('--num_episode', default=10000, type=int)  # num of episodes in training
parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  # 每50episode保存一次模型
parser.add_argument('--load', default=False, type=bool)  # 训练前是否读取模型
parser.add_argument('--render_interval', default=50, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)  # 动作向量的噪声扰动的方差
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_frame', default=200, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()
Tensor = FloatTensor

EPSILON = 0
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 40
MEMORY_CAPACITY = 100
BATCH_SIZE = 80
LR = 0.01
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)
env = env.unwrapped

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = 25
action_dim = 5

max_action = float(5)
min_Val = torch.tensor(1e-7).float().to(device)

directory = './exp' + script_name + args.env_name + './'


class Replay_buffer():
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X.reshape(-1), copy=False))
            y.append(np.array(Y.reshape(-1), copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        a = F.softmax(a, dim=1)
        return a

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self, state_dim, action_dim, max_action):
        self.lr_actor = args.learning_rate
        self.lr_critic = args.learning_rate
        self.betas = (0.9, 0.999)
        # 6个网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor, betas=self.betas)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr_critic, betas=self.betas)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr_critic, betas=self.betas)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, epoch):

        for i in range(epoch):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            next_action = self.actor_target(next_state)
            noise = torch.ones_like(next_action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (next_action + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # 延迟更新策略函数
            if i % args.policy_delay == 0:
                # 计算策略函数的损失
                actor_loss = - self.critic_1(state, self.actor(
                    state)).mean()  # 随着更新的进行Q1和Q2两个网络，将会变得越来越像。所以用Q1还是Q2，还是两者都用，对于actor的问题不大。

                # 优化策略函数
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.actor_target.state_dict(), directory + 'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory + 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory + 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory + 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory + 'critic_2_target.pth')
        print("====================================")
        print("模型保存...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("加载模型...")
        print("====================================")


def main():
    agent = TD3(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for epoch in range(args.test_episode):  # 50
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = np.random.choice([0, 1, 2, 3, 4], p=action.ravel())
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                env.render()
                if done or t == args.max_frame - 1:
                    print("测试次数： {}, 奖励 ： {:0.2f}, 步数： {}".format(epoch, ep_r, t))
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print(" 收集经验 开始训练...")
        print("====================================")
        if args.load: agent.load()
        df = pd.DataFrame([['bushu', 'jiangli', 'yongshi'],
                           ])
        df.to_csv('td3new.csv', mode='a', header=False)
        for epoch in range(args.num_episode):
            e1 = time.time()
            state = env.reset()
            for t in range(args.max_frame):
                action = agent.select_action(state)
                action2 = np.random.choice([0, 1, 2, 3, 4], p=action.ravel())
                next_state, reward, done, info = env.step(action2)
                ep_r += reward
                if args.render and epoch >= args.render_interval:
                    env.render()
                agent.memory.push((state, next_state, action, reward, np.float64(done)))
                state = next_state

                if len(agent.memory.storage) >= args.capacity - 1:
                    agent.update(args.epoch)

                if done or t == args.max_frame - 1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=epoch)
                    if epoch % args.print_log == 0:
                        print("Ep_i {}, 奖励 is {:0.2f}, the step is {}".format(epoch, ep_r, t))
                    e2 = time.time()
                    t3 = e2-e1
                    df = pd.DataFrame([[epoch, ep_r, t3], ])
                    df.to_csv('td3new.csv', mode='a', header=False)
                    ep_r = 0
                    break
            if epoch % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
