import gym
from gym import spaces, error, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
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
from trajectory import Trajectory


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, visualization = True):

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

        self.visualization = visualization

        # problem parameters from file
        self.n_agents = 1
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.quadrotors = []
        self.viewer = None

        self.trajSelect = np.zeros(3)
        # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,
        #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
        #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
        #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
        #                                 12: pos_waypoint_arrived
        self.trajSelect[0] = 3
        # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
        self.trajSelect[1] = 2
        # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
        self.trajSelect[2] = 1

        self.v_average = 1.0
        self.period_denum = 1.0
        self.dtau = 1e-3

        # intitialize state matrices
        self.total_states = np.zeros((self.n_agents, self.nx_system))
        self.agent_features = np.zeros((self.n_agents, self.n_action))
        self.diff_target = np.zeros((self.n_agents, self.n_action))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        self.max_action = 50.0
        self.gain = 1.0  # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(
            self.n_action * self.n_agents,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        # intitialize grid information
        self.x_lim = 20  # grid x limit
        self.y_lim = 20  # grid y limit
        self.z_lim = 15  # grid z limit
        self.res = 1.0  # resolution for grids
        self.out_shape = 164  # width and height for uncertainty matrix

        X, Y, Z = np.mgrid[-self.x_lim:self.x_lim + 0.1:self.res, -
                           self.y_lim:self.y_lim + 0.1:self.res, 0:self.z_lim + 0.1:self.res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T
        #self.uncertainty_values = np.ones((self.uncertainty_grids.shape[0], ))
        self.uncertainty_values = np.random.uniform(
            low=0.95, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))

        self.fig = None
        self.line1 = None
        self.counter = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, ref_pos):
        #self.nu = 1
        self.agent_targets = np.reshape(ref_pos, (self.n_agents, self.n_action))
        self.fail_check = np.zeros(self.n_agents)
        eps = 0.25
        done = False
        traj_list = []
        drone_crash = False
        reward = 0

        # self.agent_targets = np.copy(self.agent_pos_goal)

        # for i in range(self.n_agents):
        #     print ("Agent State: ", self.quadrotors[i].state)

        for i in range(self.n_agents):
            xd, yd, zd = self.agent_targets[i][0], self.agent_targets[i][1], self.agent_targets[i][2]
            pos0 = [self.quadrotors[i].state[0],
                    self.quadrotors[i].state[1], self.quadrotors[i].state[2]]
            posf = [xd, yd, zd]
            yaw0 = self.quadrotors[i].state[5]
            yawf = 0.
            time_list = np.hstack((0., 20)).astype(float)
            waypoint_list = np.vstack((pos0, posf)).astype(float)
            yaw_list = np.hstack((yaw0, yawf)).astype(float)

            newTraj = Trajectory(
                self.trajSelect, self.quadrotors[i].state, time_list, waypoint_list, yaw_list, v_average=self.v_average)
            Tf = newTraj.t_wps[1]
            flight_period = Tf / self.period_denum
            Waypoint_length = flight_period // self.dtau
            t_list = np.linspace(0, flight_period, num=int(Waypoint_length))

            print("Initial X:{0:.3}, Y:{1:.3}, Z:{2:.3} of Agent {3}".format(
                pos0[0], pos0[1], pos0[2], i+1))
            print("Target X:{0:.3}, Y:{1:.3}, Z:{2:.3} in {3:.3} s. ".format(
                xd, yd, zd, newTraj.t_wps[1]))

            for ind, t_current in enumerate(t_list):
                pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(
                    t_current, self.dtau, self.quadrotors[i].state)

                # self.vel_sum += (self.quad.state[6]**2+self.quad.state[7]**2+self.quad.state[8]**2)

                xd, yd, zd = pos_des[0], pos_des[1], pos_des[2]
                xd_dot, yd_dot, zd_dot = vel_des[0], vel_des[1], vel_des[2]
                xd_ddot, yd_ddot, zd_ddot = acc_des[0], acc_des[1], acc_des[2]

                # xd_dddot = (xd_ddot - self.xd_ddot_pr) / self.dtau
                # yd_dddot = (yd_ddot - self.yd_ddot_pr) / self.dtau
                # xd_ddddot = (xd_dddot - self.xd_dddot_pr) / self.dtau
                # yd_ddddot = (yd_dddot - self.yd_dddot_pr) / self.dtau

                psid = euler_des[2]

                # psid_dot = (psid - self.psid_pr) / self.dtau
                # psid_ddot = (psid_dot - self.psid_dot_pr) / self.dtau

                # current_traj = [xd, yd, zd, xd_dot, yd_dot, zd_dot, xd_ddot, yd_ddot, zd_ddot,
                #                 xd_dddot, yd_dddot, xd_ddddot, yd_ddddot,
                #                 psid, psid_dot, psid_ddot]

                current_traj = [xd, yd, zd, xd_dot, yd_dot, zd_dot, xd_ddot, yd_ddot, zd_ddot,
                                0, 0, 0, 0,
                                psid, 0, 0]

                self.fail_check[i] = self.quadrotors[i].simulate(current_traj)

                if self.fail_check[i]:
                    drone_crash = True
                    print("Drone {0} has crashed!".format(i))
                    break

                current_pos = [self.quadrotors[i].state[0],
                               self.quadrotors[i].state[1], self.quadrotors[i].state[2]]

                reward -= 0.00025

                if ind % 100 == 0:
                    if self.visualization:
                        self.visualize()
                    differences = current_pos - self.uncertainty_grids
                    distances = np.sum(differences * differences, axis=1)
                    min_ind = np.argmin(distances)
                    reward += 5*self.uncertainty_values[min_ind]
                    self.grid_visits[min_ind] += 1
                    self.uncertainty_values[min_ind] = np.clip(
                        np.exp(-self.grid_visits[min_ind] / 2), 1e-3, 1.0)

                    # print ("current_pos: ", current_pos)
                    # print ("closest grid: ", self.uncertainty_grids[min_ind])

            self.diff_target[i][:] = self.quadrotors[i].state[0:3] - self.agent_targets[i][:]

            print("Final  X:{0:.3}, Y:{1:.3}, Z:{2:.3}, Reward:{3:.3} for agent {4}: ".format(
                self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2], reward, i))

        if drone_crash:
            done = True
            reward = -1e4
        # elif np.sum(self.diff_target[:, 0]) <= eps and np.sum(self.diff_target[:, 1]) <= eps and np.sum(self.diff_target[:, 2]) <= eps:
        #     reward += 10

        return self._get_obs(), reward, done, {}

    def _get_obs(self):

        for i in range(self.n_agents):
            self.agent_features[i, 0] = self.quadrotors[i].state[0] - \
                0 * self.agent_pos_goal[i, 0]
            self.agent_features[i, 1] = self.quadrotors[i].state[1] - \
                0 * self.agent_pos_goal[i, 1]
            self.agent_features[i, 2] = self.quadrotors[i].state[2] - \
                0 * self.agent_pos_goal[i, 2]

        uncertainty_mat = np.reshape(
            self.uncertainty_values, (1, 1, self.out_shape, self.out_shape))
        # if self.dynamic:
        #     state_network = self.get_connectivity(self.x)
        # else:
        #     state_network = self.a_net

        # return (state_values, state_network)
        return self.agent_features, uncertainty_mat

    def instant_cost(self):  # sum of differences in velocities
        cost = 0.
        for i in range(self.n_agents):
            for j in range(3):
                cost += (self.agent_targets[i, j] -
                         self.quadrotors[i].state[j])**2

        return -np.sqrt(cost)

    def reset(self):
        x = np.zeros((self.n_agents, 2 * self.n_action))
        self.agent_features = np.zeros((self.n_agents, self.n_action))
        self.quadrotors = []
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        self.counter = 0
        self.agent_pos_goal = np.zeros((self.n_agents, self.n_action))
        self.agent_pos_start = np.zeros((self.n_agents, self.n_action))
        self.uncertainty_values = np.random.uniform(
            low=0.95, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))

        eps = 5.0

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
            state0 = [x_start, y_start, z_start,
                      0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.quadrotors.append(Quadrotor(state0))

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

        # self.x = x

        # self.a_net = self.get_connectivity(self.x)

        # pdb.set_trace()
        return self._get_obs()

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

    def visualize(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim, self.x_lim, -self.y_lim, self.y_lim)

            self.drone_transform = rendering.Transform()
            fname = path.join(path.dirname(__file__), "assets/drone.png")
            self.drone = rendering.Image(fname, 3., 3.)
            self.drone.add_attr(self.drone_transform)

        self.viewer.add_onetime(self.drone)
        self.drone_transform.set_translation(self.quadrotors[0].state[0], self.quadrotors[0].state[1])
        self.drone_transform.set_rotation(self.quadrotors[0].state[5])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


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
