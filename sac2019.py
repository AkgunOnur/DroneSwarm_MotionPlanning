import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# import pybullet_envs
import gym
import random
import numpy as np
from collections import deque
import copy
from torch.autograd import Variable

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class BasicBuffer:

    def __init__(self, max_size=1_000_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        uncertainty_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_uncertainty_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state_uncertainty, action, reward, next_state_uncertainty, done = experience
            state, uncertainty = state_uncertainty
            next_state, next_uncertainty = next_state_uncertainty

            state_batch.append(state)
            uncertainty_batch.append(uncertainty)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_uncertainty_batch.append(next_uncertainty)
            done_batch.append(done)

        return (state_batch, uncertainty_batch, action_batch, reward_batch, next_state_batch, next_uncertainty_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, n_agents, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        # self.linear1 = nn.Linear(num_inputs + num_actions, 400)
        # self.linear2 = nn.Linear(400, 300)
        # self.linear3 = nn.Linear(300, 1)

        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64, 12)
        self.linear2 = nn.Linear(3 + (n_agents - 1), 3 + (n_agents - 1))
        self.linear3 = nn.Linear(18 + (n_agents - 1), 3)
        self.linear4 = nn.Linear(15 + (n_agents - 1), 3)

    def forward(self, state, uncertainty, action, N=1):
        out1 = F.relu(self.conv1(uncertainty))
        out2 = F.relu(self.pool(out1))
        out3 = F.relu(self.conv2(out2))
        #print("uncertain_conv3_output: ", out3)
        out4 = F.relu(self.pool(out3))
        #print("uncertain_conv4_output: ", out4)
        # out5 = F.relu(self.conv2(out4))
        # print ("uncertain_conv5_output: " , out5)
        out5 = out4.view(N, -1)
        # Fully Connected Layer for Uncertainty Map
        uncertain_fcn = F.relu(self.linear1(out5))
        #print("uncertain_fcn_output: ", uncertain_fcn)

        # Fully Connected Layer for Features
        features_fcn = F.relu(self.linear2(state))

        # Uncertainty FCN and Features FCN are combined
        x = torch.cat((features_fcn, uncertain_fcn), 1)

        x = torch.cat([x, action], 1)
        x = self.linear3(x)

        return x


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, n_agents, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.linear1 = nn.Linear(num_inputs, 400)
        # self.linear2 = nn.Linear(400, 300)

        self.mean_linear = nn.Linear(300, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(300, num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64, 12)
        self.linear2 = nn.Linear(3 + (n_agents - 1), 3 + (n_agents - 1))
        self.linear3 = nn.Linear(15 + (n_agents - 1), 3)
        self.linear4 = nn.Linear(15 + (n_agents - 1), 3)

        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        # torch.nn.init.xavier_uniform(self.conv3.weight)
        # torch.nn.init.xavier_uniform(self.linear1.weight)
        # torch.nn.init.xavier_uniform(self.linear2.weight)
        # torch.nn.init.xavier_uniform(self.linear3.weight)
        # torch.nn.init.xavier_uniform(self.linear4.weight)

    def forward(self, states, uncertainty, N=1):
        out1 = F.relu(self.conv1(uncertainty))
        out2 = F.relu(self.pool(out1))
        out3 = F.relu(self.conv2(out2))
        #print("uncertain_conv3_output: ", out3)
        out4 = F.relu(self.pool(out3))
        #print("uncertain_conv4_output: ", out4)
        # out5 = F.relu(self.conv2(out4))
        # print ("uncertain_conv5_output: " , out5)
        out5 = out4.view(N, -1)
        #print("uncertain_conv5_output: ", out5.shape)

        # Fully Connected Layer for Uncertainty Map
        uncertain_fcn = F.relu(self.linear1(out5))
        #print("uncertain_fcn_output: ", uncertain_fcn)

        # Fully Connected Layer for Features
        features_fcn = F.relu(self.linear2(states))
        # print ("states: ", states.size())
        # print ("uncertain_fcn: ", uncertain_fcn.size())
        # print ("features_fcn: ", features_fcn.size())

        # Uncertainty FCN and Features FCN are combined
        x = torch.cat((features_fcn, uncertain_fcn), 1)

        mu = F.tanh(self.linear3(x))
        sigma = F.tanh(self.linear4(x))

        return mu, sigma

    # def forward(self, state):
    #     x = F.relu(self.linear1(state))
    #     x = F.relu(self.linear2(x))

    #     mean = self.mean_linear(x)
    #     log_std = self.log_std_linear(x)
    #     log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # return mean, log_std

    def sample(self, next_states, next_uncertainty, N=1, epsilon=1e-5):
        mean, log_std = self.forward(next_states, next_uncertainty, N)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi

    def deterministic(self, state):
        action, _ = self.forward(state)
        return action


class SACAgent:

    def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen, n_agents):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        # initialize networks
        self.q_net1 = SoftQNetwork(
            env.observation_space.shape[0], env.action_space.shape[0], n_agents).to(self.device)
        self.q_net2 = SoftQNetwork(
            env.observation_space.shape[0], env.action_space.shape[0], n_agents).to(self.device)
        self.target_q_net1 = SoftQNetwork(
            env.observation_space.shape[0], env.action_space.shape[0], n_agents).to(self.device)
        self.target_q_net2 = SoftQNetwork(
            env.observation_space.shape[0], env.action_space.shape[0], n_agents).to(self.device)
        self.policy_net = PolicyNetwork(
            env.observation_space.shape[0], env.action_space.shape[0], n_agents).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=policy_lr)

        # entropy temperature
        self.alpha = alpha
        self.target_entropy = - \
            torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, drone_state, uncertainty):
        drone_state = Variable(torch.FloatTensor(drone_state)).to(self.device)
        uncertainty = Variable(torch.FloatTensor(uncertainty)).to(self.device)
        #state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(drone_state, uncertainty)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action2(action)

    def get_action_deterministic(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy_net.deterministic(state)
        action = torch.tanh(action)
        action = action.cpu().detach().squeeze(0).numpy()
        return action

    def rescale_action2(self, action):
        # (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0
        return action * 2.0

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0

    def train(self, iterations, batch_size):
        for _ in range(iterations):
            states, uncertainties, actions, rewards, next_states, next_uncertainties, dones = self.replay_buffer.sample(
                batch_size)

            states = torch.FloatTensor(states).to(self.device)
            uncertainties = torch.FloatTensor(uncertainties).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            next_uncertainties = torch.FloatTensor(
                next_uncertainties).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            dones = dones.view(dones.size(0), -1)

            states = states.resize_((batch_size, states.size()[-1]))
            next_states = next_states.resize_(
                (batch_size, next_states.size()[-1]))
            uncertainties = uncertainties.resize_(
                (batch_size, 1, uncertainties.size()[-1], uncertainties.size()[-1]))
            next_uncertainties = next_uncertainties.resize_(
                (batch_size, 1, next_uncertainties.size()[-1], next_uncertainties.size()[-1]))

            # print ("states: ", states.size())
            # print ("uncertainties: ", uncertainties.size())
            # print ("actions: ", actions.size())
            # print ("rewards: ", rewards.size())
            # print ("next_states: ", next_states.size())
            # print ("rewards: ", rewards.size())
            # print ("next_states: ", next_states.size())
            # print ("next_uncertainties: ", next_uncertainties.size())
            # print ("dones: ", dones.size())

            next_actions, next_log_pi = self.policy_net.sample(
                next_states, next_uncertainties, N=batch_size)
            next_q1 = self.target_q_net1(
                next_states, next_uncertainties, next_actions.to(self.device), N=batch_size)
            next_q2 = self.target_q_net2(
                next_states, next_uncertainties, next_actions.to(self.device), N=batch_size)
            next_q_target = torch.min(
                next_q1, next_q2) - self.alpha * next_log_pi
            expected_q = rewards + (1 - dones) * self.gamma * next_q_target

            # q loss
            curr_q1 = self.q_net1.forward(
                states, uncertainties, actions, N=batch_size)
            curr_q2 = self.q_net2.forward(
                states, uncertainties, actions, N=batch_size)
            q1_loss = F.mse_loss(curr_q1, expected_q.detach())
            q2_loss = F.mse_loss(curr_q2, expected_q.detach())

            # update q networks
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            # delayed update for policy network and target q networks
            new_actions, log_pi = self.policy_net.sample(
                states, uncertainties, N=batch_size)
            if self.update_step % self.delay_step == 0:
                min_q = torch.min(
                    self.q_net1.forward(
                        states, uncertainties, actions, N=batch_size),
                    self.q_net2.forward(
                        states, uncertainties, actions, N=batch_size)
                )
                policy_loss = (self.alpha * log_pi - min_q).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # target networks
                for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                    target_param.data.copy_(
                        self.tau * param + (1 - self.tau) * target_param)

                for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                    target_param.data.copy_(
                        self.tau * param + (1 - self.tau) * target_param)

            # update temperature
            alpha_loss = (self.log_alpha * (-log_pi -
                                            self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            self.update_step += 1

    def save_checkpoint(self, policy_net, q_net):
        torch.save(self.q_net1.state_dict(), f'{q_net}_q_net_1.pt')
        torch.save(self.q_net2.state_dict(), f'{q_net}_q_net_2.pt')
        torch.save(self.policy_net.state_dict(), f'{policy_net}_policy.pt')

    def load_checkpoint(self, policy_net, q_net):
        self.q_net1 = torch.load(
            f'{q_net}_q_net_1.pt', map_location=self.device)
        self.q_net2 = torch.load(
            f'{q_net}_q_net_2.pt', map_location=self.device)
        self.policy_net = torch.load(
            f'{policy_net}_policy.pt', map_location=self.device)
