from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
from AstarF import *
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

        self.misc_arr = np.zeros((self.N_iteration, self.args.nagents))
        # self.state_arr = np.zeros((self.N_iteration, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        # self.next_state_arr = np.zeros((self.N_iteration, self.args.nagents, (self.args.nagents+1)*5+2, self.env.out_shape, self.env.out_shape))
        self.state_arr = np.zeros((self.N_iteration, self.args.nagents, self.args.nagents*3 + 1))
        self.next_state_arr = np.zeros((self.N_iteration, self.args.nagents, self.args.nagents*3 + 1))
        self.uncertainty_arr = np.zeros((self.N_iteration, self.env.out_shape, self.env.out_shape))
        self.next_uncertainty_arr = np.zeros((self.N_iteration, self.env.out_shape, self.env.out_shape))
        self.action_arr = np.zeros((self.N_iteration, self.args.nagents))
        self.action_out_arr = np.zeros((self.N_iteration, self.args.nagents, self.env.n_action))
        self.value_arr = np.zeros((self.N_iteration, self.args.nagents))
        self.episode_mask_arr = np.zeros((self.N_iteration, self.args.nagents))
        self.episode_mini_mask_arr = np.zeros((self.N_iteration, self.args.nagents))
        self.reward_arr = np.zeros((self.N_iteration, self.args.nagents))


    def get_episode(self, epoch):
        episode = []

        stat = dict()
        info = dict()
        switch_t = -1

        total_obs, info = self.env.reset()
        state, uncertainty_map = total_obs

        final_step = self.N_iteration
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.N_iteration):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            state = torch.DoubleTensor(state)
            state = state.view(self.args.nagents, self.args.nagents*3 + 1)
            uncertainty_map = torch.DoubleTensor(uncertainty_map)
            uncertainty_map = uncertainty_map.view(1,1,self.env.out_shape, self.env.out_shape)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=1)

                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, uncertainty_map, info)
                

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, uncertainty_map, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            total_obs, reward, done, info = self.env.step(action, t, self.is_centralized)
            next_state, uncertainty_map = total_obs

            if (t+1) % 20 == 0:
                print ("T-Eps/Iter: {0}/{1}, Actions: {2}, Rewards: {3}, Agent status: {4}".format(epoch+1, t+1, action[0], reward, self.env.agent_status))

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

            # trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            # episode.append(trans)

            self.misc_arr[t,:] = np.copy(misc['alive_mask'])
            self.state_arr[t,:,:] = np.copy(state)
            self.next_state_arr[t,:,:] = np.copy(next_state)
            self.action_arr[t,:] = np.copy(action)
            self.action_out_arr[t,:, :] = np.copy(action_out[0].detach().numpy())
            self.value_arr[t, :] = np.copy(value.detach().numpy()).ravel()
            self.episode_mask_arr[t, :] = np.copy(episode_mask)
            self.episode_mini_mask_arr[t, :] = np.copy(episode_mini_mask)
            self.reward_arr[t,:] = np.copy(reward)
            
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

        batch = self.state_arr[0:final_step], self.action_arr[0:final_step], self.action_out_arr[0:final_step], self.value_arr[0:final_step], self.episode_mask_arr[0:final_step], self.episode_mini_mask_arr[0:final_step], self.next_state_arr[0:final_step], self.reward_arr[0:final_step], self.misc_arr[0:final_step]
        
        return batch, stat

    def run_batch(self, epoch):
        self.stats = dict()
        self.stats['num_episodes'] = 0

        state_total_batch = np.array([]).reshape((0, self.args.nagents, (self.args.nagents)*3+1))
        next_state_total_batch = np.array([]).reshape((0, self.args.nagents, (self.args.nagents)*3+1))
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
            log_p_a = [action_out.view(-1, self.env.n_action)]
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


class Tester(Trainer):
    def __init__(self, args, policy_net, env, is_centralized = False):
        super().__init__(args, policy_net, env, is_centralized)

    def get_episode(self, epoch):
        episode = []

        stat = dict()
        info = dict()
        switch_t = -1

        total_obs, info = self.env.reset()
        state, uncertainty_map = total_obs

        final_step = self.N_iteration
        pos_list = np.zeros((3, self.N_iteration, self.env.n_agents))
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.N_iteration):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            state = torch.DoubleTensor(state)
            state = state.view(self.args.nagents, self.args.nagents*3 + 1)
            uncertainty_map = torch.DoubleTensor(uncertainty_map)
            uncertainty_map = uncertainty_map.view(1,1,self.env.out_shape, self.env.out_shape)

            # state = state.view(-1, self.args.nagents, state.size(-3), state.size(-2), state.size(-1))

            for j in range(self.env.n_agents):
                # print ("state {0}: X:{1:.3}, Y:{2:.3}, Z:{3:.3}".format(i+1, self.env.quadrotors[i].state[0], 
                #                                                 self.env.quadrotors[i].state[1],self.env.quadrotors[i].state[2] ))
                pos_list[:, t, j] = self.env.quadrotors[j].state[0:3]

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=1)

                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, uncertainty_map, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, uncertainty_map, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            total_obs, reward, done, info = self.env.step(action, t, self.is_centralized)
            next_state, uncertainty_map = total_obs
            if (t+1) % 20 == 0:
                print ("E-Eps/Iter: {0}/{1}, Actions: {2}, Rewards: {3}, Agent status: {4}".format(epoch+1, t+1, action[0], reward, self.env.agent_status))

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
            
            state = np.copy(next_state)
            
            if done:
                final_step = t + 1
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        
        return pos_list, stat

    def test_batch(self, save=True):
        batch = []
        N_epoch = 10
            
        total_pos_list = []
        for epoch in range(N_epoch):
            pos_list, stat = self.get_episode(epoch)
            if save:
                total_pos_list.append(pos_list)
                with open('agents_positions_4.pkl', 'wb') as f:
                    pickle.dump(total_pos_list, f)
            else:
                return stat

        return stat


class Planner:
    def __init__(self, env, is_centralized = False):
        np.set_printoptions(precision=2)
        self.env = env
        self.is_centralized = is_centralized
        self.agent_indices = np.zeros(self.env.n_agents)
        self.uncertainty_thresh = 0.2
        self.N_iteration = 1000


    def get_episode(self, epoch):
        stat = dict()
        astar_path = dict()
        obs_x0, obs_y0, obs_z0 = self.env.obstacle_points[0][0], self.env.obstacle_points[0][1], self.env.obstacle_points[0][2]  
        obs_x1, obs_y1, obs_z1 = self.env.obstacle_points[0][3], self.env.obstacle_points[0][4], self.env.obstacle_points[0][5]  

        self.env.reset()

        pos_list = np.zeros((3, self.N_iteration, self.env.n_agents))  

        on_the_battery_way = np.zeros(self.env.n_agents)
        agent_battery_path = {i:[] for i in range(self.env.n_agents)}

        battery_location_center = [[(battery[0] + battery[3])/2, (battery[1] + battery[4])/2, (battery[2] + battery[5])/2] 
                                    for battery in self.env.battery_points]
        agent_n_iter = np.ones(self.env.n_agents) * 3
        for t in range(self.N_iteration):
            misc = dict()
            action_list = []
            
            for agent_index in range(self.env.n_agents):
                # print ("state {0}: X:{1:.3}, Y:{2:.3}, Z:{3:.3}".format(i+1, self.env.quadrotors[i].state[0], 
                #                                                 self.env.quadrotors[i].state[1],self.env.quadrotors[i].state[2] ))
                start_pos = self.env.quadrotors[agent_index].state[0:3]
                pos_list[:, t, agent_index] = self.env.quadrotors[agent_index].state[0:3]


                if self.env.battery_status[agent_index] < self.env.battery_critical_level and len(agent_battery_path[agent_index]) == 0:
                    on_the_battery_way[agent_index] = 1
                    distance_to_battery = [np.sum((start_pos-battery)**2) for battery in battery_location_center]
                    sorted_indices = np.argsort(distance_to_battery)
                    # print ("start pos: ", start_pos)
                    for index in sorted_indices:
                        if (start_pos[0] < self.env.obstacle_points[0][0] < self.env.battery_points[index][0]) or \
                            (start_pos[0] > self.env.obstacle_points[0][0] > self.env.battery_points[index][0]):
                            # print ("not allowed: ", battery_location_center[index])
                            continue
                        else:
                            final_pos = battery_location_center[index]
                            final_pos[2] = start_pos[2]
                            # print ("allowed: ", final_pos)
                            break

                    # index = np.argmin(np.sum(np.subtract(start_pos, self.env.battery_positions)**2,axis=1))
                    # closest_battery_pos = self.env.battery_positions[index]
                    # closest_battery_index = self.env.get_closest_grid(closest_battery_pos)
                    # final_pos = self.env.uncertainty_grids[closest_battery_index]
                    # final_pos = [final_pos[0], final_pos[1], start_pos[2]] # z locations should be the same
                    path = astar_drone(start_pos, final_pos, self.env)
                    agent_battery_path[agent_index] = path[1:]
                    print ("Battery status of agent {0} is below critical level. It's heading to the battery station!".format(agent_index+1))
                elif self.env.battery_status[agent_index] >= self.env.battery_critical_level and on_the_battery_way[agent_index] == 1:
                    on_the_battery_way[agent_index] = 0
                    agent_battery_path[agent_index] = []
                    print ("Agent {0} is fully recharged!".format(agent_index+1))


                if len(agent_battery_path[agent_index]) > 0:
                    action = agent_battery_path[agent_index][0][1]
                    agent_battery_path[agent_index].pop(0) 
                    print ("Agent {0} is executing the A* path. Action: {1}".format(agent_index+1, action))
                else:

                    if self.env.agent_is_stuck[agent_index] == 1.0:
                        pos_coef = 10.0
                        escape_index = -1
                        start_pos = np.array(self.env.quadrotors[agent_index].state[0:3])
                    
                        while (escape_index == -1):
                            escape_pos = np.array([np.random.uniform(-pos_coef, pos_coef), np.random.uniform(-pos_coef, pos_coef), start_pos[2]])
                            escape_index = self.env.get_closest_grid(escape_pos)
                            
                            if escape_index in self.env.obstacle_indices:
                                escape_index = -1

                        escape_pos = self.env.uncertainty_grids[escape_index]
                        print ("Agent {0} is stuck. Its position: {1} and heading to a random position: {2}".format(agent_index+1, start_pos, escape_pos))
                        path = astar_drone(start_pos, escape_pos, self.env)
                        agent_battery_path[agent_index] = path[1:]
                        

                    else:
                        neighbor_grids = self.env.get_neighbor_grids(agent_index)
                        action_index_pairs = dict(neighbor_grids)
                        
                        for key in neighbor_grids.keys():
                            neighbor_pos = self.env.uncertainty_grids[neighbor_grids[key]]
                            if (neighbor_pos[0] >= obs_x0 and neighbor_pos[0] <= obs_x1) and (neighbor_pos[1] >= obs_y0 and neighbor_pos[1] <= obs_y1):
                                # print ("Key: {0} is in the obstacle list: {1}".format(neighbor_grids[key], np.array(neighbor_pos)))
                                del action_index_pairs[key]
                            elif neighbor_grids[key] in self.env.agent_pos_index :
                                # print ("Key: {0} is in the agent list".format(neighbor_grids[key]))
                                del action_index_pairs[key]
                            # elif neighbor_grids[key] in self.env.battery_indices :
                            #     # print ("Key: {0} is in the agent list".format(neighbor_grids[key]))
                            #     del action_index_pairs[key]
                            # else:
                            #     if self.env.uncertainty_values[neighbor_grids[key]] < self.uncertainty_thresh:
                            #         # print ("Key: {0} is under the given uncertainty threshold".format(neighbor_grids[key]))
                            #         del action_index_pairs[key]


                        allowable_actions = [*action_index_pairs]
                        if len(allowable_actions) > 0:
                            open_grids =  [neighbor_grids[act] for act in allowable_actions]
                            non_battery_grids = np.setdiff1d(open_grids, self.env.battery_indices)
                            # print ("prob values: ", self.env.uncertainty_values[open_grids])
                            probabilities = self.env.uncertainty_values[open_grids]
                            probabilities[open_grids == non_battery_grids] = np.clip(probabilities[open_grids == non_battery_grids], 0.3, 1.0)
                            normalized_prob = probabilities / np.sum(probabilities)
                            action = np.random.choice(allowable_actions, p=normalized_prob)
                        else:
                            action = -1


                

                action_list.append(action)

                # print (action_index_pairs)
                # print (allowable_actions)
                # print ("Agent: {0} random action from the list: {1}".format(agent_index, action))
            
            reward, done = self.env.step(action_list, t, self.is_centralized)
            print ("E-Episode/Iteration: {0}/{1}, Actions: {2}, Rewards: {3}".format(epoch+1, t+1, action_list, reward))


            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward
                        
            if done:
                break

        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        
        return pos_list, stat

    def test_batch(self, save=True):
        batch = []
        N_epoch = 5
            
        total_pos_list = []
        for epoch in range(N_epoch):
            pos_list, stat = self.get_episode(epoch)
            if save:
                total_pos_list.append(pos_list)
                with open('./agents_position/agents_positions_planner.pkl', 'wb') as f:
                    pickle.dump(total_pos_list, f)
            else:
                return stat

        return stat
