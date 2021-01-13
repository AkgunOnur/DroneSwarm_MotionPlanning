### Import Libraries ###
import gym
import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
from make_g import build_graph
import torch.optim as optim
import dgl.function as fn
import math

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import matplotlib.pyplot as plt
from policy import Net
#from linear_policy import Net
from utils import *
from quadrotor_formation import QuadrotorFormation

import os
import datetime
import warnings
warnings.filterwarnings("ignore")

# Define Policy, optimizer and environment
policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = QuadrotorFormation(visualization=True)

# generate model folder if not exists
if not os.path.exists('./models'):
    os.makedirs('./models')

# Determine file name and save model parameters
filename = "single_agent.pt"
torch.save(policy.state_dict(), './models/%s' % filename)


### main function ###
def main(episodes):
    running_reward = 10
    plotting_rew = []
    mean_reward_pr = -np.Inf

    # Start simulation for all episodes
    for episode in range(episodes):
        reward_over_eps = []
        pos_target = np.array([[0., 0., 0.]])
        agent_obs = env.reset()  # Reset environment and record the starting state
        #g = build_graph(env)  # build graph
        done = False
        # Episode loop
        state, uncertainty_mat = agent_obs
        for time in range(200):
            # if episode%50==0:
            #g = build_graph(env)
            #print("state: ", state)
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
            
            # Print goals for all quads
            

            agent_obs, reward, done, _ = env.step(ref_pos)
            state, uncertainty_mat = agent_obs
            reward_over_eps.append(reward)
            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        mean_reward = np.mean(reward_over_eps)
        if(episode >= 3):
            update_policy(policy, optimizer)

        if episode % 1 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage reward over episode: {:.2f}'.format(
                episode, time, mean_reward))

        # Save policy for every 5000 episodes
        if mean_reward > mean_reward_pr:
            mean_reward_pr = mean_reward
            torch.save(policy.state_dict(), './models/%s' % filename)

        plotting_rew.append(np.mean(reward_over_eps))

    # Saving and plotting training results
    # pdb.set_trace()
    np.savetxt('Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %
               (env.n_agents), plotting_rew)
    fig = plt.figure()
    x = np.linspace(0, len(plotting_rew), len(plotting_rew))
    plt.plot(x, plotting_rew)
    plt.savefig('Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %
                (env.n_agents))
    plt.show()

    # pdb.set_trace()


if __name__ == "__main__":
    episodes = 5000  # Determining number of episodes
    main(episodes)  # Calling main function
