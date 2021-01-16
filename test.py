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
from policy import Net
from make_g import build_graph
from utils import *
from quadrotor_formation import QuadrotorFormation

import os 
import datetime 
import warnings
warnings.filterwarnings("ignore")

policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = QuadrotorFormation(visualization=True)
filename = './models/single_agent.pt'
policy.load_state_dict(torch.load(filename))

#pdb.set_trace()

def main():
	test_episodes = 5
	plotting_rew = []
	
	for episode in range(test_episodes):
		reward_over_eps = []
		pos_target = np.array([[0., 0., 0.]])
		agent_obs = env.reset()  # Reset environment and record the starting state
		#g = build_graph(env)  # build graph
		done = False

		state, uncertainty_mat = agent_obs
		for time in range(200):
			#g = build_graph(env)
			action = select_action(state, uncertainty_mat,  policy)

			action = action.numpy()
			#print ("\n action_0: ", action)
			print ("\n Episode: {0}, Iteration: {1}".format(episode+1, time+1))
			print("Action X: {0:.4}, Y: {1:.4}, Z: {2:.4}".format(
				action[0][0], action[0][1], action[0][2]))
			pos_target = pos_target + action
			ref_pos = np.reshape(pos_target, [-1])
			# Step through environment using chosen action
			ref_pos[0] = np.clip(ref_pos[0], -env.x_lim, env.x_lim)
			ref_pos[1] = np.clip(ref_pos[1], -env.y_lim, env.y_lim)
			ref_pos[2] = np.clip(ref_pos[2], 0.5, env.z_lim)

			agent_obs, reward, done, _ = env.step(ref_pos)
			state, uncertainty_mat = agent_obs
			reward_over_eps.append(reward)
			# Save reward
			policy.reward_episode.append(reward)
			if done:
				break

		mean_reward = np.mean(reward_over_eps)
		print('Episode {}\tLast length: {:5d}\tAverage reward over episode: {:.2f}'.format(
                episode, time, mean_reward))
		

		plotting_rew.append(np.mean(reward_over_eps))
	
	np.savetxt('Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %(env.n_agents), plotting_rew)
	fig = plt.figure()
	x = np.linspace(0,len(plotting_rew),len(plotting_rew))
	plt.plot(x,plotting_rew)
	plt.savefig('Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %(env.n_agents))
	plt.show()



if __name__ == '__main__':
	main()