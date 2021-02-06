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
warnings.filterwarnings("ignore")

n_agents = 1
gamma = 0.99
tau = 0.005
alpha = 0.2
a_lr = 1e-3
q_lr = 1e-3
p_lr = 1e-3
buffer_maxlen = 1_000_000

env = QuadrotorFormation(n_agents = n_agents, visualization=True)
filename = './models/actor_3_200_policy.pt'
policy = SAC(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)
# policy.load_state_dict(torch.load(filename))

policy.policy_net.load_state_dict(torch.load(filename))

# pdb.set_trace()


def main():
    test_episodes = 2
    plotting_rew = []

    for episode in range(test_episodes):
        pos_target = np.array([[0., 0., 0.]])
        reward_over_eps = []
        agent_obs = env.reset()
        episode_timesteps = 0
        for time in range(500):
            # if total_timesteps < start_timesteps:
            #     action = env.action_space.sample()
            # else:
            drone_state, uncertainty_mat = agent_obs
            action = policy.get_action(drone_state, uncertainty_mat)

            #action = action.numpy()
            print("\n Episode: {0}, Iteration: {1}".format(
                episode + 1, time + 1))
            print("Action X: {0:.4}, Y: {1:.4}, Z: {2:.4}".format(
                action[0], action[1], action[2]))

            pos_target = pos_target + action
            ref_pos = np.reshape(pos_target, [-1])
            # Step through environment using chosen action
            ref_pos[0] = np.clip(ref_pos[0], -env.x_lim, env.x_lim)
            ref_pos[1] = np.clip(ref_pos[1], -env.y_lim, env.y_lim)
            ref_pos[2] = np.clip(ref_pos[2], 0.5, env.z_lim)

            agent_new_obs, reward, done, _ = env.step(ref_pos)
            reward_over_eps.append(reward)

            drone_new_state, new_uncertainty_mat = agent_new_obs

            policy.replay_buffer.add(
                (drone_state, uncertainty_mat), action, reward, (drone_new_state, new_uncertainty_mat), done)

            agent_obs = agent_new_obs

            episode_timesteps += 1
            #total_timesteps += 1

            if done:
                break

        # Used to determine when the environment is solved.
        mean_reward = np.mean(reward_over_eps)

        print('Episode {}\tLast length: {:5d}\tAverage reward over episode: {:.2f}'.format(
            episode, time, mean_reward))

        plotting_rew.append(np.mean(reward_over_eps))

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
