from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import time

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


class Trainer(object):
    def __init__(self, args, policy_net, env, is_centralized):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.print_count = 0
        self.is_centralized = is_centralized
        self.optimizer = optim.RMSprop(policy_net.parameters(),
                                       lr=args.lrate, alpha=0.97, eps=1e-6)
        #self.optimizer = optim.Adam(policy_net.parameters(), lr = args.lrate)
        self.params = [p for p in self.policy_net.parameters()]

    def get_episode(self, epoch):
        episode = []
        stat = dict()
        info = dict()
        switch_t = -1

        surveillance_rate_list = []
        episode_rate_list = []

        if self.args.scenario == 'planning':
            total_obs, info = self.env.reset()
            state, uncertainty_map = total_obs

        elif self.args.scenario == 'predator':
            state = self.env.reset()           

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        agent_pos_list = []
        bot_pos_list = []
        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            if self.args.scenario == 'predator':
                state = torch.DoubleTensor(state)
                state = state.unsqueeze(0)

            elif self.args.scenario == 'planning':
                state = torch.DoubleTensor(state)
                state = state.view(self.args.nagents,
                                   self.args.nagents * 3 + 1)
                uncertainty_map = torch.DoubleTensor(uncertainty_map)
                uncertainty_map = uncertainty_map.view(
                    1, 1, self.env.out_shape, self.env.out_shape)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=1)
                
                x = [state, prev_hid]
                if self.args.scenario == 'predator':
                    action_out, value, prev_hid = self.policy_net(
                        x, None, info)
                elif self.args.scenario == 'planning':
                    action_out, value, prev_hid = self.policy_net(
                        x, uncertainty_map, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.args.scenario == 'predator':
                    action_out, value = self.policy_net(x, None, info)
                elif self.args.scenario == 'planning':
                    action_out, value, prev_hid = self.policy_net(
                        x, uncertainty_map, info)

            if self.args.scenario == 'predator':
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)

                next_state, reward, done, info, agent_pos, bot_pos = self.env.step(
                    action, t, False)

            elif self.args.scenario == 'planning':
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                total_obs, reward, done, info, agent_pos, surveillance_rate = self.env.step(
                    action[0], t, self.is_centralized)
                next_state, uncertainty_map = total_obs

            agent_pos_list.append(np.array(agent_pos))
            if self.args.scenario == 'predator':
                bot_pos_list.append(np.array(bot_pos))

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(
                    self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get(
                    'comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm'] = stat.get(
                        'enemy_comm', 0) + info['comm_action'][self.args.nfriendly:]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']
            stat['reward'] = stat.get('reward', 0) + \
                reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get(
                    'enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            trans = Transition(state, action, action_out, value,
                               episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state

            if self.args.scenario == 'planning':
                surveillance_rate_list.append(surveillance_rate)
            
                if t % 1 == 0:
                    # Initial call to print 0% progress
                    printProgressBar(100*surveillance_rate, 100, prefix = 'Episode ' + str(epoch) + ": ", suffix = 'Surveillance Rate', length = 50)
                    # for i, item in enumerate(items):
                    #     # Do stuff...
                    #     time.sleep(0.1)
                    #     # Update Progress Bar
                    #     printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

            if done:
                break

        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if self.args.scenario == "predator":
            n_agent = len(bot_pos_list[-1])
            score = 0
            for idx in range(n_agent):
                if bot_pos_list[-1][idx][2] == 0.:
                    score += 1.0
            
            total_score = score/n_agent*100.0
            if self.print_count == epoch:
                print("EPISODE:",epoch,"SUCCESS RATE = %",total_score)
                self.print_count += 1
            else:
                pass

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(
                reward=episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + \
                reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get(
                    'enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)

        if self.args.scenario == 'predator':
            self.env.close()
            return (episode, stat), agent_pos_list, bot_pos_list
        elif self.args.scenario == 'planning':
            return (episode, stat), agent_pos_list, np.mean(surveillance_rate_list)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]
        alive_masks = torch.Tensor(np.concatenate(
            [item['alive_mask'] for item in batch.misc])).view(-1)

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
            coop_returns[i] = rewards[i] + self.args.gamma * \
                prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * \
                prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

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
            log_prob = normal_log_density(
                actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i])
                       for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)
            # print("log_p_a",log_p_a)
            #print("actions shape", actions.shape)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        # print("actions",actions)
        # print(actions.shape)
        # print("log_prob",log_prob)
        # print(log_prob.shape)

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

        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            if self.args.scenario == 'predator':
                epx, agent_pos_list, bot_pos_list = self.get_episode(epoch)
            elif self.args.scenario == 'planning':
                epx, agent_pos_list, mean_surv_rate = self.get_episode(epoch)
            episode, episode_stat = epx
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))

        if self.args.scenario == "planning":
            return batch, self.stats, mean_surv_rate
        elif self.args.scenario == "predator":
            return batch, self.stats

    def test_run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0

        if self.args.scenario == 'predator':
            _, agent_pos, bot_pos = self.get_episode(epoch)
            self.stats['num_episodes'] += 1
            return batch, self.stats, agent_pos, bot_pos

        elif self.args.scenario == 'planning':
            _, agent_pos, mean_surv_rate = self.get_episode(epoch)
            self.stats['num_episodes'] += 1
            return batch, self.stats, agent_pos
        # print(bot_pos_list)
        # print(agent_pos_list)
        # ep_agent_pos.append(agent_pos_list)
        # ep_bot_pos.append(bot_pos_list)

    # only used when nprocesses=1

    def train_batch(self, epoch):
        if self.args.scenario == "planning":
            batch, stat, mean_surv_rate = self.run_batch(epoch)
        elif self.args.scenario == "predator":
            batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        if self.args.scenario == "planning":
            return stat, mean_surv_rate
        elif self.args.scenario == "predator":
            return stat

    def test_batch(self, epoch):
        if self.args.scenario == 'predator':
            stat, _, agent_pos, bot_pos = self.test_run_batch(epoch)
            return stat, agent_pos, bot_pos
        elif self.args.scenario == 'planning':
            stat, _, agent_pos = self.test_run_batch(epoch)
            return stat, agent_pos

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
