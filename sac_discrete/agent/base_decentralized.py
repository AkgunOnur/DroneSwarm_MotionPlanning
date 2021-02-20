from abc import ABC, abstractmethod
import os
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

from sac_discrete.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from sac_discrete.utils import update_params, RunningMeanStats


class BaseAgent_Decentralized(ABC):

    def __init__(self, env, n_agents=2, num_steps=100000, batch_size=128,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=200,
                 update_interval=4, target_update_interval=5,
                 use_per=False, num_eval_steps=125000, max_episode_steps=20000, max_iteration_steps=300,
                 log_interval=10, eval_interval=500, cuda=True, seed=0):
        super().__init__()

        self.env = env
        self.n_agents = n_agents
        agent_obs_shape = (self.env.N_frame*(self.env.n_agents+1)+1, self.env.out_shape, self.env.out_shape)

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        # self.device = torch.device(
        #     "cuda" if cuda and torch.cuda.is_available() else "cpu")
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = "cpu"

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = [LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=agent_obs_shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps) for i in range(self.n_agents)]
        else:
            self.memory = [ LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=agent_obs_shape,
                device=self.device, gamma=gamma, multi_step=multi_step) for i in range(self.n_agents)]

        self.model_dir = '/okyanus/users/deepdrone/Independent_DroneSwarm/DroneSwarm_MotionPlanning/models'
        self.summary_dir = '/okyanus/users/deepdrone/Independent_DroneSwarm/DroneSwarm_MotionPlanning/summary'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.learning_steps = 0
        self.best_eval_score = [-np.inf for i in range(self.n_agents)]
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
    def explore(self, agent_ind, state, device):
        pass

    @abstractmethod
    def exploit(self, agent_ind, state, device):
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
            episode_return = [0. for i in range(self.n_agents)]
            agent_obs = self.env.reset()
            done = False

            for iteration in range(self.max_iteration_steps):
                action = np.zeros(self.n_agents)
                for agent_ind in range(self.n_agents):
                    if episode < self.start_steps:
                        action[agent_ind] = self.env.action_space.sample()
                    else:
                        action[agent_ind] = self.explore(agent_ind, agent_obs, self.device)

                next_agent_obs, reward, done, _ = self.env.step(action, iteration)

                # Clip reward to [-1.0, 1.0].
                # clipped_reward = max(min(reward, 1.0), -1.0)

                # To calculate efficiently, set priority=max_priority here.
                for agent_ind in range(self.n_agents):
                    self.memory[agent_ind].append(agent_obs, action[agent_ind], reward[agent_ind], next_agent_obs, done)
                    episode_return[agent_ind] += reward[agent_ind]

                agent_obs = next_agent_obs


            if self.is_update(episode):
                self.learn()

            if episode % self.target_update_interval == 0 and episode >= self.start_steps:
                self.update_target()

            if episode % self.eval_interval == 0 and episode >= self.start_steps:
                self.evaluate()
                for agent_ind in range(self.n_agents):
                    self.save_models(os.path.join(self.model_dir, 'final'), agent_ind)


            print(f'Episode: {episode:<5}  '
                f'Return 1: {episode_return[0]:<5.1f}  '
                f'Return 2: {episode_return[1]:<5.1f}')

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1
        for agent_ind in range(self.n_agents):

            if self.use_per:
                batch, weights = self.memory[agent_ind].sample(self.batch_size)
            else:
                batch = self.memory[agent_ind].sample(self.batch_size)
                # Set priority weights to 1 when we don't use PER.
                weights = 1.

            q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
                self.calc_critic_loss(batch, weights, agent_ind)
            policy_loss, entropies = self.calc_policy_loss(batch, weights, agent_ind)
            entropy_loss = self.calc_entropy_loss(entropies, weights)

            update_params(self.q1_optim[agent_ind], q1_loss)
            update_params(self.q2_optim[agent_ind], q2_loss)
            update_params(self.policy_optim[agent_ind], policy_loss)
            update_params(self.alpha_optim[agent_ind], entropy_loss)

            self.alpha = self.log_alpha.exp()

            if self.use_per:
                self.memory[agent_ind].update_priority(errors)

    def evaluate(self):        
        agent_obs = self.env.reset()
        iteration_steps = 1
        episode_return = np.zeros(self.n_agents)
        done = False

        while iteration_steps <= self.max_iteration_steps:
            action = self.exploit(agent_obs, self.device)
            next_agent_obs, reward, done, _ = self.env.step(action, iteration_steps)
            iteration_steps += 1
            episode_return += reward
            agent_obs = next_agent_obs

        for agent_ind in range(self.n_agents):
            if episode_return[agent_ind] > self.best_eval_score[agent_ind]:
                print ("Better reward obtained for Agent {0}. The reward: {1:.3f}".format(agent_ind+1, episode_return[agent_ind]))
                self.best_eval_score[agent_ind] = episode_return[agent_ind]
                self.save_models(os.path.join(self.model_dir, 'best'), agent_ind)

                # print(f'Evaluation Mode'
                #       f'Return {agent_ind+1:<2}: {episode_return[agent_ind]:<5.1f}  ')

    @abstractmethod
    def save_models(self, save_dir, agent_ind):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # def __del__(self):
    #     self.writer.close()
