from sac2019 import SACAgent as SAC
import gym
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import math

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import matplotlib.pyplot as plt
from point_mass_formation import QuadrotorFormation

import os
import datetime
import warnings
from time import sleep
warnings.filterwarnings("ignore")
#######################

if not os.path.exists("./models"):
    os.makedirs("./models")


# env_name = "Walker2DBulletEnv-v0"
# env = gym.make(env_name)
def main():
    n_agents = 2
    N_episodes = 20000
    N_iteration = 250
    train_episode_modulo = 5
    batch_size = 50
    

    env = QuadrotorFormation(n_agents=n_agents, visualization=True)
    plotting_rew = []
    mean_reward_pr = -np.Inf

    start_timesteps = 10_000
    eval_freq = 5_000
    max_timesteps = 500_000

    total_timesteps = 0
    episode_reward = 0
    episode_num = 0
    done = False

    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    a_lr = 1e-3
    q_lr = 1e-3
    p_lr = 1e-3
    buffer_maxlen = 1_000_000

    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # env.seed(seed)

    policy_list = []

    for i in range(n_agents):
        policy = SAC(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen, n_agents)
        policy_list.append(policy)

    done = False

    for episode in range(1, N_episodes+1):
        reward_over_eps = []
        agent_obs, pos_target = env.reset()
        for time in range(1, N_iteration+1):
            # if total_timesteps < start_timesteps:
            #     action = env.action_space.sample()
            # else:
            print("\n Episode: {0}/{2}, Iteration: {1}/{3}".format(
                episode, time, N_episodes, N_iteration))

            action_list = []
            ref_pos = np.zeros((n_agents, 3))
            drone_state, conv_stack = agent_obs
            for i in range(n_agents):
                action = policy_list[i].get_action(drone_state[i,:].reshape(1,-1), conv_stack)
                action_list.append(action)
                # print("Action X: {0:.4}, Y: {1:.4}, Z: {2:.4}".format(action[0], action[1], action[2]))
                    

                pos_target[i,:] = pos_target[i,:] + action
                # ref_pos = np.reshape(pos_target, [-1])
                ref_pos[i,0] = np.clip(pos_target[i,0], -env.x_lim, env.x_lim)
                ref_pos[i,1] = np.clip(pos_target[i,1], -env.y_lim, env.y_lim)
                ref_pos[i,2] = np.clip(pos_target[i,2], 0.5, env.z_lim)


            agent_new_obs, reward_list, done, _ = env.step(ref_pos)
            reward_over_eps.append(reward_list)

            drone_new_state, new_conv_stack = agent_new_obs

            for i in range(n_agents):
                policy = policy_list[i]
                for j in range(n_agents):
                    policy.replay_buffer.add(
                        (drone_state[j,:].reshape(1,-1), conv_stack), action_list[j], reward_list[j], (drone_new_state[j,:].reshape(1,-1), new_conv_stack), done)

            agent_obs = agent_new_obs

            if done:
                break

        
        # Used to determine when the environment is solved.
        mean_reward = np.mean(reward_over_eps)
        if(episode % train_episode_modulo == 0):
            for i in range(n_agents):
                policy_list[i].train(10, batch_size)

        if episode % 1 == 0:
            print('Episode {}\tIteration: {:5d}\tAverage reward over episode: {:.2f}'.format(
                episode, time, mean_reward))

        if env.visualization:
            sleep(2.0)
            env.viewer.close()

        if mean_reward > mean_reward_pr:
            mean_reward_pr = mean_reward
            for i in range(n_agents):
                policy_list[i].save_checkpoint('models/best_actor_' + str(i+1), 'models/best_critic_' + str(i+1))
        elif episode % 500 == 0:
            for i in range(n_agents):
                policy_list[i].save_checkpoint('models/actor_' + str(i+1) + '_' + str(episode), 'models/critic_' + str(i+1) + '_' + str(episode))

        plotting_rew.append(np.mean(reward_over_eps))

    # pdb.set_trace()
    np.savetxt('Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %
               (env.n_agents), plotting_rew)
    fig = plt.figure()
    x = np.linspace(0, len(plotting_rew), len(plotting_rew))
    plt.plot(x, plotting_rew)
    plt.savefig('Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %
                (env.n_agents))
    # plt.show()


if __name__ == "__main__":
    main()  # Calling main function
