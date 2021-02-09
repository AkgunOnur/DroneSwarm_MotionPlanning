import gym
from sac2019 import SACAgent as SAC
#import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch as t
#from make_g import build_graph
import torch.optim as optim
import dgl
import dgl.function as fn
import math
import pdb
import pickle

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
import matplotlib.pyplot as plt
#from policy import Net
#from make_g import build_graph
#from utils import *
from quadrotor_formation import QuadrotorFormation

import os
import datetime
import warnings
from time import sleep

warnings.filterwarnings("ignore")

n_agents = 3
gamma = 0.99
tau = 0.005
alpha = 0.2
a_lr = 1e-3
q_lr = 1e-3
p_lr = 1e-3
buffer_maxlen = 1_000_000
policy_list = []
N_iteration = 200
N_episode = 5

env = QuadrotorFormation(n_agents = n_agents, visualization=True)

for i in range(n_agents):
    filename = './models/actor_' + str(i+1) + '_1600_policy.pt'
    policy = SAC(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen, n_agents)
    policy.policy_net.load_state_dict(torch.load(filename))
    policy_list.append(policy)


# pdb.set_trace()


def main():
    plotting_rew = []
    agent_pos_over_episodes = []

    for episode in range(N_episode):
        reward_over_eps = []
        agent_obs, pos_target = env.reset()
        episode_timesteps = 0
        agent_pos_dict = {i:[] for i in range(n_agents)}

        for time in range(N_iteration):
            # if total_timesteps < start_timesteps:
            #     action = env.action_space.sample()
            # else:

            print("\n Episode: {0}/{2}, Iteration: {1}/{3}".format(
                episode + 1, time + 1, N_episode, N_iteration))

            action_list = []
            ref_pos = np.zeros((n_agents, 3))
            drone_state, uncertainty_mat = agent_obs
            for i in range(n_agents):
                action = policy_list[i].get_action(drone_state[i,:].reshape(1,-1), uncertainty_mat)
                action_list.append(action)
                # print("Action X: {0:.4}, Y: {1:.4}, Z: {2:.4}".format(action[0], action[1], action[2]))
                    

                pos_target[i,:] = pos_target[i,:] + action
                # ref_pos = np.reshape(pos_target, [-1])
                ref_pos[i,0] = np.clip(pos_target[i,0], -env.x_lim, env.x_lim)
                ref_pos[i,1] = np.clip(pos_target[i,1], -env.y_lim, env.y_lim)
                ref_pos[i,2] = np.clip(pos_target[i,2], 0.5, env.z_lim)

            agent_new_obs, reward_list, done, agent_pos_dict = env.step(ref_pos, agent_pos_dict)
            reward_over_eps.append(reward_list)

            drone_new_state, new_uncertainty_mat = agent_new_obs

            agent_obs = agent_new_obs

            if done:
                break

        agent_pos_over_episodes.append(agent_pos_dict)
        if env.visualization:
            N_max = np.max([len(agent_pos_dict[i]) for i in agent_pos_dict.keys()])
            for j in range(N_max):
                pos_list = []
                for i in range(n_agents):
                    index = np.clip(j, 0, len(agent_pos_dict[i])-1)
                    pos_list.append(agent_pos_dict[i][index])
                env.visualize(pos_list)
                sleep(0.01)


        # Used to determine when the environment is solved.
        mean_reward = np.mean(reward_over_eps)

        print('Episode {}\tLast length: {:5d}\tAverage reward over episode: {:.2f}'.format(
            episode, time, mean_reward))

        plotting_rew.append(np.mean(reward_over_eps))

        if episode % 1 == 0:
            with open('models/agents_positions.pkl', 'wb') as f:
                pickle.dump(agent_pos_over_episodes, f)

    np.savetxt('Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %
               (env.n_agents), plotting_rew)
    fig = plt.figure()
    x = np.linspace(0, len(plotting_rew), len(plotting_rew))
    plt.plot(x, plotting_rew)
    plt.savefig('Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %
                (env.n_agents))
    plt.show()


if __name__ == '__main__':
    main()
