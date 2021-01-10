# Import libraries
import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from sklearn.neighbors import NearestNeighbors
import itertools
import random
import pdb
from quadrotor_dynamics import Quadrotor
from numpy.random import uniform

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):
    def __init__(self):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True  # if the agents are moving or not
        # normalize the adjacency matrix by the number of neighbors or not
        self.mean_pooling = False
        # self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 1
        # number of features per agent
        self.n_features = 12
        # number states per agent
        self.nx_system = self.n_features + 3
        # number of actions per agent which are desired positions and yaw angle
        self.n_action = 3
        self.x_lim = 50.0  # figure x limit
        self.y_lim = 50.0  # figure y limit

        # problem parameters from file
        self.n_agents = 4
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.quadrotors = [Quadrotor(state0)] * self.n_agents

        # intitialize state matrices
        self.total_states = np.zeros((self.n_agents, self.nx_system))
        self.agent_features = np.zeros((self.n_agents, self.n_action))
        self.diff_target = np.zeros((self.n_agents, self.n_action))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        self.max_action = 50
        # TODO - adjust if necessary - may help the NN performance (unused)
        self.gain = 1.0
        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(
            self.n_action * self.n_agents,), dtype=np.float32)

        # an unused variable
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.counter = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Environment step function
    def step(self, action):
        #self.nu = 1
        self.agent_targets = np.reshape(action, (self.n_agents, self.n_action))
        self.fail_check = np.zeros(self.n_agents)
        eps = 1.5
        done = False

        for i in range(self.n_agents):
            #     ref_traj = [xd[i], yd[i], zd[i], xd_dot[i], yd_dot[i], zd_dot[i],
            #                 xd_ddot[i], yd_ddot[i], zd_ddot[i], xd_dddot[i], yd_dddot[i],
            #                 xd_ddddot[i], yd_ddddot[i], psid[i], psid_dot[i], psid_ddot[i]]
            xd, yd, zd = self.agent_targets[i][0], self.agent_targets[i][1], self.agent_targets[i][2]
            ref_traj = [xd, yd, zd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.fail_check[i] = int(self.quadrotors[i].simulate(ref_traj))
            self.diff_target[i][:] = self.quadrotors[i].state[0:3] - \
                self.agent_targets[i][:]

        if np.sum(self.fail_check) > 0:
            done = True
            reward = -1e4
            return self._get_obs(), reward, done, {}

        if np.sum(self.diff_target[:, 0]) < eps and np.sum(self.diff_target[:, 1]) < eps and np.sum(self.diff_target[:, 2]) < eps:
            # REward???
            done = True

        return self._get_obs(), self.instant_cost(), done, {}

    def instant_cost(self):  # sum of differences in velocities
        cost = 0.
        for i in range(self.n_agents):
            for j in range(3):
                cost += (self.agent_targets[i, j] -
                         self.quadrotors[i].state[j])**2

        return -np.sqrt(cost)

    # Environment reset function
    def reset(self):
        x = np.zeros((self.n_agents, 2 * self.n_action))
        self.agent_features = np.zeros((self.n_agents, self.n_action))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        self.counter = 0
        self.agent_pos_goal = np.zeros((self.n_agents, self.n_action))
        self.agent_pos_start = np.zeros((self.n_agents, self.n_action))

        eps = 2.0

        goal_locations = [(uniform(-self.x_lim - eps, -self.x_lim + eps), uniform(-self.y_lim - eps, -self.y_lim + eps), uniform(5 - eps, 5 + eps)),
                          (uniform(-self.x_lim - eps, -self.x_lim + eps),
                           uniform(self.y_lim - eps, self.y_lim + eps), uniform(5 - eps, 5 + eps)),
                          (uniform(self.x_lim - eps, self.x_lim + eps), uniform(self.y_lim -
                                                                                eps, self.y_lim + eps), uniform(5 - eps, 5 + eps)),
                          (uniform(self.x_lim - eps, self.x_lim + eps), uniform(-self.y_lim - eps, -self.y_lim + eps), uniform(5 - eps, 5 + eps))]

        for i in range(0, self.n_agents):
            x_target, y_target, z_target = goal_locations[i]
            self.agent_pos_goal[i, :] = [x_target, y_target, z_target]

        # scheme :
        # space all agents in a frontier that looks like -> .'.
        # this means everyone has x goal position separated by two units.
        # and y goal position varies
        # let the number of agents be odd. for kosher reasons.
        # hack, say n agents = 3. Then placer_x is = -2
        self.placer_x = (self.n_agents / 2) * 2 * (-1)

        ##########declare start positions############
        for i in range(0, self.n_agents):
            x_start = self.placer_x + 2 * i
            y_start = uniform(-1., 0)
            z_start = uniform(0, 2.)

            self.agent_pos_start[i, :] = [x_start, y_start, z_start]

        x[:, 0] = self.agent_pos_start[:, 0]
        x[:, 1] = self.agent_pos_start[:, 1]
        x[:, 2] = self.agent_pos_start[:, 2]

        x[:, 3] = self.agent_pos_goal[:, 0]
        x[:, 4] = self.agent_pos_goal[:, 1]
        x[:, 5] = self.agent_pos_goal[:, 2]

        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        init_action = self.agent_pos_goal

        # pdb.set_trace()
        return self._get_obs()

    def _get_obs(self):

        for i in range(self.n_agents):
            self.agent_features[i, 0] = self.quadrotors[i].state[0] - \
                self.agent_pos_goal[i, 0]
            self.agent_features[i, 1] = self.quadrotors[i].state[1] - \
                self.agent_pos_goal[i, 1]
            self.agent_features[i, 2] = self.quadrotors[i].state[2] - \
                self.agent_pos_goal[i, 2]

        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        # return (state_values, state_network)
        return self.agent_features

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:3], (self.n_agents, 3, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
                                 np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_connectivity(self, x):

        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:, 3:6])
            a_net = np.array(neigh.kneighbors_graph(
                mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            # TODO or axis=0? Is the mean in the correct direction?
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors

        return a_net

    # Environment render function
    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        edge_list = []
        adj_matrix = self.get_connectivity(self.x)

        num_ag = self.n_agents
        for i in range(0, num_ag):
            for j in range(0, num_ag):
                if adj_matrix[i][j] > 0:
                    edge_list.append((i, j))

        edge_list = np.array(edge_list)

        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # line1, = ax.plot(self.x[:, 0], self.x[:, 1], linestyle='-', color='y',markerfacecolor='blue', marker='o')  # Returns a tuple of line objects, thus the comma
            # Returns a tuple of line objects, thus the comma
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')
            #line1 = ax.plot(x[edge_list.T], y[edge_list.T], linestyle='-', color='y',markerfacecolor='red', marker='o')

            ax.plot([0], [0], 'kx')
            ax.plot(self.agent_pos_start[:, 0],
                    self.agent_pos_start[:, 1], 'kx')
            ax.plot(self.agent_pos_goal[:, 0], self.agent_pos_goal[:, 1], 'rx')

            # plt.ylim(-15* self.r_max, 15.0 * self.r_max)
            # plt.xlim(-10.0 * self.r_max, 10.0 * self.r_max)
            plt.ylim(-self.y_lim, self.y_lim)
            plt.xlim(-self.x_lim, self.x_lim)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('%d Robots Formation' % self.n_agents)
            self.fig = fig
            self.line1 = line1

            plt.gca().legend(('Robots'))
        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
