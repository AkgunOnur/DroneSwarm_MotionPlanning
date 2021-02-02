from sac2019 import SACAgent as SAC
import numpy as np
import os
import torch
import gym
# import pybullet_envs
from gym import wrappers
#######################
### Import Libraries ###
import gym
#import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
#from make_g import build_graph
import torch.optim as optim
import dgl.function as fn
import math

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import matplotlib.pyplot as plt
#from policy import Net
#from linear_policy import Net
#from utils import *
from quadrotor_formation import QuadrotorFormation

import os
import datetime
import warnings
warnings.filterwarnings("ignore")
#######################

if not os.path.exists("./models"):
    os.makedirs("./models")


# def evaluate_policy(policy, eval_episodes=10):
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             action = policy.get_action(obs)
#             obs, reward, done, _ = env.step(action)
#             avg_reward += reward
#     avg_reward /= eval_episodes
#     print("\n------------------------------------------")
#     print(f"SAMPLE: Evaluation Step: {avg_reward}")
#     print("------------------------------------------\n")


# def evaluate_policy_deterministic(policy, eval_episodes=10):
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             action = policy.get_action_deterministic(obs)
#             obs, reward, done, _ = env.step(action)
#             avg_reward += reward
#     avg_reward /= eval_episodes
#     print("\n------------------------------------------")
#     print(f"DETERMINISTIC: Evaluation Step: {avg_reward}")
#     print("------------------------------------------\n")


# env_name = "Walker2DBulletEnv-v0"
# env = gym.make(env_name)
def main(episodes):
    env = QuadrotorFormation(visualization=False)
    plotting_rew = []
    mean_reward_pr = -np.Inf

    start_timesteps = 10_000
    eval_freq = 5_000
    max_timesteps = 500_000
    batch_size = 100
    max_episode_steps = 200  # env._max_episode_steps

    total_timesteps = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    done = False

    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    a_lr = 1e-3
    q_lr = 1e-3
    p_lr = 1e-3
    buffer_maxlen = 1_000_000

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    # env.seed(seed)

    policy = SAC(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

    done = False

    for episode in range(episodes):
        pos_target = np.array([[0., 0., 0.]])
        reward_over_eps = []
        agent_obs = env.reset()
        episode_timesteps = 0
        for time in range(200):
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
            total_timesteps += 1

            if done:
                break

        # Used to determine when the environment is solved.
        mean_reward = np.mean(reward_over_eps)
        if((episode + 1) % 2 == 0):
            policy.train(1, batch_size)

        if episode % 1 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage reward over episode: {:.2f}'.format(
                episode, time, mean_reward))

        # Save policy for every 5000 episodes
        if mean_reward > mean_reward_pr:
            mean_reward_pr = mean_reward
            policy.save_checkpoint('models/actor', 'models/critic')
        elif episode % 100 == 0:
            policy.save_checkpoint(
                'models/actor' + str(episode), 'models/critic' + str(episode))

        plotting_rew.append(np.mean(reward_over_eps))

    # pdb.set_trace()
    np.savetxt('Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %
               (env.n_agents), plotting_rew)
    fig = plt.figure()
    x = np.linspace(0, len(plotting_rew), len(plotting_rew))
    plt.plot(x, plotting_rew)
    plt.savefig('Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %
                (env.n_agents))
    plt.show()

    # if total_timesteps >= start_timesteps:
    #         policy.train(episode_timesteps, batch_size)
    #     print("Total Timesteps: {} Episode Timesteps {} Episode Num: {} Reward: {}".format(
    #         total_timesteps, episode_timesteps, episode_num, episode_reward))
    #     obs = env.reset()
    #     episode_reward = 0
    #     episode_timesteps = 0
    #     episode_num += 1

    # if total_timesteps % eval_freq == 0:
    #     evaluate_policy(policy)
    #     evaluate_policy_deterministic(policy)
    #     policy.save_checkpoint('models/actor', 'models/critic')


if __name__ == "__main__":
    episodes = 5000  # Determining number of episodes
    main(episodes)  # Calling main function
