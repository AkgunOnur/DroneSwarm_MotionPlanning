from abc import ABC, abstractmethod
import os
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

from sac_discrete.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from sac_discrete.utils import update_params, RunningMeanStats


class BaseAgent(ABC):

    def __init__(self, env, num_steps=100000, batch_size=128,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=200,
                 update_interval=4, target_update_interval=5,
                 use_per=False, num_eval_steps=125000, max_episode_steps=20000, max_iteration_steps=300,
                 log_interval=10, eval_interval=500, cuda=True, seed=0):
        super().__init__()

        self.env = env
        agent_obs_shape = (self.env.N_frame*(self.env.n_agents+1)+1, self.env.out_shape, self.env.out_shape)

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        
        self.device = "cpu"

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=agent_obs_shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=agent_obs_shape,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.model_dir = './models'
        self.summary_dir = './summary'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.learning_steps = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.max_iteration_steps = max_iteration_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval


    def is_update(self, episode):
        return episode % self.update_interval == 0\
            and episode >= self.start_steps

    @abstractmethod
    def explore(self, state, device):
        pass

    @abstractmethod
    def exploit(self, state, device):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    def train_episode(self):
        
        for episode in range(self.max_episode_steps):
            episode_return = 0.
            agent_obs = self.env.reset()
            done = False

            for iteration in range(self.max_iteration_steps):

                if episode < self.start_steps :
                    action = self.env.action_space.sample()
                else:
                    action = self.explore(agent_obs, self.device)

                next_agent_obs, reward, done, _ = self.env.step(action, iteration)

                # Clip reward to [-1.0, 1.0].
                # clipped_reward = max(min(reward, 1.0), -1.0)

                # To calculate efficiently, set priority=max_priority here.
                self.memory.append(agent_obs, action, reward, next_agent_obs, done)

                episode_return += reward
                agent_obs = next_agent_obs


            if self.is_update(episode):
                self.learn()

            if episode % self.target_update_interval and episode >= self.start_steps == 0:
                self.update_target()

            if episode % self.eval_interval and episode >= self.start_steps == 0:
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

            # We log running mean of training rewards.
            self.train_return.append(episode_return)

            # if episode % self.log_interval == 0:
            #     self.writer.add_scalar(
            #         'reward/train', self.train_return.get(), episode)

            print(f'Episode: {episode:<4}  '
                f'Return: {episode_return:<5.1f}')

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        # if self.learning_steps % self.log_interval == 0:
        #     self.writer.add_scalar(
        #         'loss/Q1', q1_loss.detach().item(),
        #         self.learning_steps)
        #     self.writer.add_scalar(
        #         'loss/Q2', q2_loss.detach().item(),
        #         self.learning_steps)
        #     self.writer.add_scalar(
        #         'loss/policy', policy_loss.detach().item(),
        #         self.learning_steps)
        #     self.writer.add_scalar(
        #         'loss/alpha', entropy_loss.detach().item(),
        #         self.learning_steps)
        #     self.writer.add_scalar(
        #         'stats/alpha', self.alpha.detach().item(),
        #         self.learning_steps)
        #     self.writer.add_scalar(
        #         'stats/mean_Q1', mean_q1, self.learning_steps)
        #     self.writer.add_scalar(
        #         'stats/mean_Q2', mean_q2, self.learning_steps)
        #     self.writer.add_scalar(
        #         'stats/entropy', entropies.detach().mean().item(),
        #         self.learning_steps)

    def evaluate(self):        
        agent_obs = self.env.reset()
        iteration_steps = 1
        episode_return = 0.0
        done = False

        while iteration_steps <= self.max_iteration_steps:
            action = self.exploit(agent_obs, self.device)
            next_agent_obs, reward, done, _ = self.env.step(action, iteration_steps)
            iteration_steps += 1
            episode_return += reward
            agent_obs = next_agent_obs

        if episode_return > self.best_eval_score:
            self.best_eval_score = episode_return
            self.save_models(os.path.join(self.model_dir, 'best'))
            print(f'Evaluation mode - Better reward: {episode_return:<5.1f}')

        # self.writer.add_scalar(
        #     'reward/test', episode_return)
        # print('-' * 60)
        # print(f'return: {episode_return:<5.1f}')
        # print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # def __del__(self):
    #     self.writer.close()
