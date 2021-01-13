import gym
import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from make_g import build_graph
import torch.optim as optim
import dgl
import dgl.function as fn
import math
import pdb

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input 164x164
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64, 12)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(15, 3)
        self.linear4 = nn.Linear(15, 3)

        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.linear1.weight)
        torch.nn.init.xavier_uniform(self.linear2.weight)
        torch.nn.init.xavier_uniform(self.linear3.weight)
        torch.nn.init.xavier_uniform(self.linear4.weight)

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = 0.99

    def forward(self, features, uncertainty):
        out1 = F.relu(self.conv1(uncertainty))
        out2 = F.relu(self.pool(out1))
        out3 = F.relu(self.conv2(out2))
        #print("uncertain_conv3_output: ", out3)
        out4 = F.relu(self.pool(out3))
        #print("uncertain_conv4_output: ", out4)
        # out5 = F.relu(self.conv2(out4))
        # print ("uncertain_conv5_output: " , out5)
        out5 = out4.view(1, -1)
        #print("uncertain_conv5_output: ", out5.shape)

        # Fully Connected Layer for Uncertainty Map
        uncertain_fcn = F.relu(self.linear1(out5))
        #print("uncertain_fcn_output: ", uncertain_fcn)

        # Fully Connected Layer for Features
        features_fcn = F.relu(self.linear2(features))

        # Uncertainty FCN and Features FCN are combined
        x = torch.cat((features_fcn, uncertain_fcn), 1)

        mu = F.tanh(self.linear3(x))
        sigma = F.tanh(self.linear4(x))

        # # Repeating uncertainty feature (1x36) for each agent (n_agentx36)
        # out5 = out5.repeat(N,1)
        # # print ("out: ", out5.size())

        # # First graph output (n_agentx16)
        # x = self.gcn1(g, features)
        # # print ("x: ", x.size())

        # # Concatenate uncertainty output and graph output (n_agentx52)
        # y = torch.cat((x, out5), 1)
        # # print ("y: ", y.size())
        # #x = F.relu(self.l1(features))
        # #mu = F.relu(self.l2(x))
        # #sigma = F.relu(self.l3(x))

        # # Last graph output (n_agentx3)
        # mu = self.gcn2(g, y)
        # # print ("mu: ", mu.size())
        # sigma = self.gcn2_(g,y)
        return mu, sigma
