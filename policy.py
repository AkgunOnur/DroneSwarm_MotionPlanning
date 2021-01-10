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

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

###############################################################################
# We then define the node UDF for ``apply_nodes``, which is a fully-connected layer:

class NodeApplyModule(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		h = self.activation(h)
		return {'h' : h}

###############################################################################
# We then proceed to define the GCN module. A GCN layer essentially performs
# message passing on all the nodes then applies the `NodeApplyModule`. Note
# that we omitted the dropout in the paper for simplicity.

class GCN(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(GCN, self).__init__()
		self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(gcn_msg, gcn_reduce)
		g.apply_nodes(func=self.apply_mod)
		return g.ndata.pop('h')

###############################################################################
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch.  We can initialize GCN like any ``nn.Module``. For example,
# let's define a simple neural network consisting of two GCN layers. Suppose we
# are training the classifier for the cora dataset (the input feature size is
# 1433 and the number of classes is 7).

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# input 164x164
		self.conv = nn.Conv2d(in_channels=1, out_channels= 1, kernel_size= 7, stride=2)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		#self.l1 = nn.Linear(2,64,bias=False)
		#self.l2 = nn.Linear(64,2,bias=False)
		#self.l3 = nn.Linear(64,2,bias=False)

		self.gcn1 = GCN(3, 16, t.tanh)
		#self.gcn11 = GCN(16, 16, t.tanh)
		#self.gcn111 = GCN(64, 32, F.relu)
		self.gcn2 = GCN(52, 3, t.tanh)
		self.gcn2_ = GCN(52,3,t.tanh)

		#self.gcn2 = GCN(16, 2, t.tanh)
		#self.gcn2_ = GCN(16,2,t.tanh)


		self.policy_history = Variable(torch.Tensor())
		self.reward_episode = []
		# Overall reward and loss history
		self.reward_history = []
		self.loss_history = []
		self.gamma = 0.99

	def forward(self, g, features, uncertainty):
		N = features.shape[0]
		# Convolution part
		out1 = F.relu(self.conv(uncertainty))
		out2 = F.relu(self.pool(out1)) 
		out3 = F.relu(self.conv(out2))
		out4 = F.relu(self.pool(out3)) 
		out5 = F.relu(self.conv(out3)) 
		out5 = out5.view(1, -1)
		# print ("out: ", out5.size())
		
		# Repeating uncertainty feature (1x36) for each agent (n_agentx36)
		out5 = out5.repeat(N,1)
		# print ("out: ", out5.size())

		# First graph output (n_agentx16)
		x = self.gcn1(g, features)
		# print ("x: ", x.size())

		# Concatenate uncertainty output and graph output (n_agentx52)
		y = torch.cat((x, out5), 1)
		# print ("y: ", y.size())
		#x = F.relu(self.l1(features))
		#mu = F.relu(self.l2(x))
		#sigma = F.relu(self.l3(x))

		# Last graph output (n_agentx3)
		mu = self.gcn2(g, y)
		# print ("mu: ", mu.size())
		sigma = self.gcn2_(g,y)
		return mu, sigma

