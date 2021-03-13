import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from sac_discrete.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory

# Hyper Parameters
BATCH_SIZE = 128
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 20    # target update frequency
N_ACTIONS = 2
N_STATES = 294
memory_size = 100000
gamma = 0.99
multi_step = 1
n_agents = 2
N_frame = 5
out_shape = 82
agent_obs_shape = (N_frame * (n_agents + 1) + 1, out_shape, out_shape)
device = "cpu"

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        num_channels = N_frame * (n_agents + 1) + 1
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 3, 3, 2),        
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  
            nn.Conv2d(3, 6, 3, 1),         
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  
            nn.Conv2d(6, 6, 3, 1),         
            nn.ReLU(),
            Flatten()
        ).apply(initialize_weights_he)

        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.net(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = LazyMultiStepMemory(
                                        capacity=memory_size,
                                        state_shape=agent_obs_shape,
                                        device=device, gamma=gamma, multi_step=multi_step)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = batch

        # q_eval w.r.t the action in experience
        
        q_eval = self.eval_net(states).gather(1, actions.long())  # shape (batch, 1)
        q_next = self.target_net(next_states).detach()     # detach from graph, don't backpropagate
        q_target = rewards + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print ("Learn function is called!")

    def save_models(self, save_dir, episode_number):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.eval_net.state_dict(), os.path.join(save_dir, 'policy_' + str(episode_number) + '.pth'))
        torch.save(self.target_net.state_dict(), os.path.join(save_dir, 'target_net_' + str(episode_number) + '.pth'))