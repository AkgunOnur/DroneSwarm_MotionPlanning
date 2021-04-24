from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import pickle

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env, is_centralized = False):
        np.set_printoptions(precision=2)
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.is_centralized = is_centralized
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.num_inputs = 294
        self.N_iteration = 1000


    def get_episode(self, epoch):
        episode = []
        
        misc_arr = np.zeros((self.N_iteration, self.args.nagents))
        state_arr = np.zeros((self.N_iteration, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        next_state_arr = np.zeros((self.N_iteration, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        action_arr = np.zeros((self.N_iteration, self.args.nagents))
        action_out_arr = np.zeros((self.N_iteration, self.args.nagents, self.env.n_action))
        value_arr = np.zeros((self.N_iteration, self.args.nagents))
        episode_mask_arr = np.zeros((self.N_iteration, self.args.nagents))
        episode_mini_mask_arr = np.zeros((self.N_iteration, self.args.nagents))
        reward_arr = np.zeros((self.N_iteration, self.args.nagents))

        stat = dict()
        info = dict()
        switch_t = -1

        reset_args = getargspec(self.env.reset).args
        total_obs, info = self.env.reset()
        state, battery_status = total_obs
        should_display = self.display and self.last_step

        final_step = self.N_iteration

        # if should_display:
        #     self.env.display()
        

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.N_iteration):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            state = torch.DoubleTensor(state)
            state = state.view(self.args.nagents, state.size(-3), state.size(-2), state.size(-1))
            battery_status = torch.DoubleTensor(battery_status)
            battery_status = battery_status.view(self.args.nagents, -1)
            # state = state.view(-1, self.args.nagents, state.size(-3), state.size(-2), state.size(-1))

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=1)

                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, battery_status, info)
                

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, battery_status, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            total_obs, reward, done, info = self.env.step(action, t, self.is_centralized)
            next_state, battery_status = total_obs
            print ("T-Episode/Iteration: {0}/{1}, Actions: {2}, Rewards: {3}".format(epoch+1, t+1, action[0], reward))

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # print ("misc: ", misc)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            # if should_display:
            #     self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            
            episode.append(trans)
            misc_arr[t,:] = np.copy(misc['alive_mask'])
            state_arr[t,:,:,:,:] = np.copy(state)
            next_state_arr[t,:,:,:,:] = np.copy(next_state)
            action_arr[t,:] = np.copy(action)
            action_out_arr[t,:, :] = np.copy(action_out[0].detach().numpy())
            value_arr[t, :] = np.copy(value.detach().numpy()).ravel()
            episode_mask_arr[t, :] = np.copy(episode_mask)
            episode_mini_mask_arr[t, :] = np.copy(episode_mini_mask)
            reward_arr[t,:] = np.copy(reward)
            
            state = np.copy(next_state)
            
            if done:
                final_step = t + 1
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        # if hasattr(self.env, 'reward_terminal'):
        #     reward = self.env.reward_terminal()
        #     # We are not multiplying in case of reward terminal with alive agent
        #     # If terminal reward is masked environment should do
        #     # reward = reward * misc['alive_mask']

        #     episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
        #     stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
        #     if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
        #         stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        # if hasattr(self.env, 'get_stat'):
        #     merge_stat(self.env.get_stat(), stat)

        # print ("state[0:final_step]: ", state[0:final_step])
        # print ("action[0:final_step]: ", action[0:final_step])
        # print ("action_out[0:final_step]: ", action_out[0:final_step])
        # print ("episode_mask[0:final_step]: ", episode_mask[0:final_step])
        # print ("episode_mini_mask[0:final_step]: ", episode_mini_mask[0:final_step])
        # print ("reward[0:final_step]: ", reward[0:final_step])
        # print ("misc[0:final_step]: ", misc[0:final_step])

        batch = state_arr[0:final_step], action_arr[0:final_step], action_out_arr[0:final_step], value_arr[0:final_step], episode_mask_arr[0:final_step], episode_mini_mask_arr[0:final_step], next_state_arr[0:final_step], reward_arr[0:final_step], misc_arr[0:final_step]
        
        return batch, stat

    def run_batch(self, epoch):
        self.stats = dict()
        self.stats['num_episodes'] = 0

        state_total_batch = np.array([]).reshape((0, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        next_state_total_batch = np.array([]).reshape((0, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        action_total_batch = np.array([]).reshape((0, self.args.nagents))
        action_out_total_batch = np.array([]).reshape((0, self.args.nagents, self.env.n_action))
        value_total_batch = np.array([]).reshape((0, self.args.nagents))
        episode_mask_total_batch = np.array([]).reshape((0, self.args.nagents))
        episode_mini_mask_total_batch = np.array([]).reshape((0, self.args.nagents))
        reward_total_batch = np.array([]).reshape((0, self.args.nagents))
        misc_total_batch = np.array([]).reshape((0, self.args.nagents))        

        while state_total_batch.shape[0] < self.args.batch_size:
            episode, episode_stat = self.get_episode(epoch)
            state_batch, action_batch, action_out_batch, value_batch, episode_mask_batch, \
            episode_mini_mask_batch, next_state_batch, reward_batch, misc_batch = episode

            state_total_batch = np.vstack([state_total_batch, state_batch])
            next_state_total_batch = np.vstack([next_state_total_batch, next_state_batch])
            action_total_batch = np.vstack([action_total_batch, action_batch])
            action_out_total_batch = np.vstack([action_out_total_batch, action_out_batch])
            value_total_batch = np.vstack([value_total_batch, value_batch])
            episode_mask_total_batch = np.vstack([episode_mask_total_batch, episode_mask_batch])
            episode_mini_mask_total_batch = np.vstack([episode_mini_mask_total_batch, episode_mini_mask_batch])
            reward_total_batch = np.vstack([reward_total_batch, reward_batch])
            misc_total_batch = np.vstack([misc_total_batch, misc_batch])
            
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1

            
            
        self.stats['num_steps'] = state_batch.shape[0]
        batch = state_total_batch, next_state_total_batch, action_total_batch, action_out_total_batch, value_total_batch, episode_mask_total_batch, episode_mini_mask_total_batch, reward_total_batch, misc_total_batch
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat, self.env.uncertainty_values

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = 1 #self.args.dim_actions

        states, next_states, actions, action_out, values, episode_masks, episode_mini_masks, rewards, miscs = batch

        n = self.args.nagents
        batch_size = states.shape[0]

        rewards = torch.Tensor(rewards)
        episode_masks = torch.Tensor(episode_masks)
        episode_mini_masks = torch.Tensor(episode_mini_masks)
        actions = torch.Tensor(actions)
        values = torch.tensor(values, requires_grad=True)
        action_out = torch.Tensor(action_out)
        # actions = actions.view(-1, n, dim_actions)
        # print ("actions: ", actions.size())

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        

        # can't do batch forward.
        # values = torch.cat(values, dim=0)
        # action_out = [torch.cat(a, dim=0) for a in action_out]

        # alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in miscs])).view(-1)
        alive_masks = torch.Tensor(miscs).view(-1)
        # print ("alive_masks: ", alive_masks)


        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()
        
        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            # action_out0:  torch.Size([20, 3, 5])
            # action_out1:  torch.Size([20, 3, 2])
            # log_p_a0:  torch.Size([60, 5])
            # log_p_a1:  torch.Size([60, 2])
            # actions:  torch.Size([60, 2])
            # actions[:, i]:  torch.Size([60])
            # actions[:, i].long().unsqueeze(1):  torch.Size([60, 1])
            # log_probs[i]:  torch.Size([60, 5])
            # actions[:, i]:  torch.Size([60])
            # actions[:, i].long().unsqueeze(1):  torch.Size([60, 1])
            # log_probs[i]:  torch.Size([60, 2])
            # print ("action_out: ", action_out.size())
            # log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            log_p_a = [action_out.view(-1, 6)]
            # print ("log_p_a: ", log_p_a[0].size())
            # print ("log_p_a: ", log_p_a)
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                # print ("actions: ", actions.size())
                log_prob = multinomials_log_density(actions, log_p_a)

        

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        # print ("loss: ", loss)
        loss.backward()

        return stat

    

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


class Tester(object):
    def __init__(self, args, policy_net, env, is_centralized = False):
        np.set_printoptions(precision=2)
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.is_centralized = is_centralized
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.num_inputs = 294
        self.N_iteration = 1000


    def get_episode(self, epoch):
        episode = []
        
        misc_arr = np.zeros((self.N_iteration, self.args.nagents))
        state_arr = np.zeros((self.N_iteration, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        next_state_arr = np.zeros((self.N_iteration, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        action_arr = np.zeros((self.N_iteration, self.args.nagents))
        action_out_arr = np.zeros((self.N_iteration, self.args.nagents, self.env.n_action))
        value_arr = np.zeros((self.N_iteration, self.args.nagents))
        episode_mask_arr = np.zeros((self.N_iteration, self.args.nagents))
        episode_mini_mask_arr = np.zeros((self.N_iteration, self.args.nagents))
        reward_arr = np.zeros((self.N_iteration, self.args.nagents))

        stat = dict()
        info = dict()
        switch_t = -1

        reset_args = getargspec(self.env.reset).args
        total_obs, info = self.env.reset()
        state, battery_status = total_obs
        should_display = self.display and self.last_step

        final_step = self.N_iteration

        # if should_display:
        #     self.env.display()
        

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.N_iteration):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            state = torch.DoubleTensor(state)
            state = state.view(self.args.nagents, state.size(-3), state.size(-2), state.size(-1))
            battery_status = torch.DoubleTensor(battery_status)
            battery_status = battery_status.view(self.args.nagents, -1)
            # state = state.view(-1, self.args.nagents, state.size(-3), state.size(-2), state.size(-1))

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=1)

                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, battery_status, info)
                

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, battery_status, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            total_obs, reward, done, info = self.env.step(action, t, self.is_centralized)
            next_state, battery_status = total_obs
            print ("E-Episode/Iteration: {0}/{1}, Actions: {2}, Rewards: {3}".format(epoch+1, t+1, action[0], reward))

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # print ("misc: ", misc)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            # if should_display:
            #     self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            
            episode.append(trans)
            misc_arr[t,:] = np.copy(misc['alive_mask'])
            state_arr[t,:,:,:,:] = np.copy(state)
            next_state_arr[t,:,:,:,:] = np.copy(next_state)
            action_arr[t,:] = np.copy(action)
            action_out_arr[t,:, :] = np.copy(action_out[0].detach().numpy())
            value_arr[t, :] = np.copy(value.detach().numpy()).ravel()
            episode_mask_arr[t, :] = np.copy(episode_mask)
            episode_mini_mask_arr[t, :] = np.copy(episode_mini_mask)
            reward_arr[t,:] = np.copy(reward)
            
            state = np.copy(next_state)
            
            if done:
                final_step = t + 1
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        batch = state_arr[0:final_step], action_arr[0:final_step], action_out_arr[0:final_step], value_arr[0:final_step], episode_mask_arr[0:final_step], episode_mini_mask_arr[0:final_step], next_state_arr[0:final_step], reward_arr[0:final_step], misc_arr[0:final_step]
        
        return batch, stat

    def test_batch(self, save=True):
        batch = []
        N_epoch = 1
            
        total_pos_list = []
        for epoch in range(N_epoch):
            episode, stat = self.get_episode(epoch)
            # if save:
            #     total_pos_list.append(pos_list)
            #     with open('agents_positions.pkl', 'wb') as f:
            #         pickle.dump(total_pos_list, f)
            # else:
            #     return stat

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)