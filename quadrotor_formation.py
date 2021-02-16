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
from time import sleep
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import




font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=1, visualization=True):

        self.X_list = []
        self.Y_list = []
        self.Z_list = []

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
        self.n_agents = n_agents
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

        self.v_average = 0.75
        self.period_denum = 1.0
        self.dtau = 1e-3

        self.xdot_d = np.zeros((self.n_agents, 3))
        self.xdotdot_d = np.zeros((self.n_agents, 3))

        # intitialize state matrices
        self.total_states = np.zeros((self.n_agents, self.nx_system))
        self.agent_features = np.zeros((self.n_agents, self.n_action + 3*(self.n_agents - 1)))
        self.diff_target = np.zeros((self.n_agents, self.n_action))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        self.max_action = 20.0
        self.gain = 1.0  # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(
            self.n_action * self.n_agents,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        # intitialize grid information
        self.x_lim = 20  # grid x limit
        self.y_lim = 20  # grid y limit
        self.z_lim = 6  # grid z limit
        self.res = 1.0  # resolution for grids
        self.out_shape = 82  # width and height for uncertainty matrix
        self.dist = 5.0  # distance threshold

        X, Y, Z = np.mgrid[-self.x_lim:self.x_lim + 0.1:self.res, -
                           self.y_lim:self.y_lim + 0.1:self.res, 0:self.z_lim + 0.1:2*self.res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T
        #self.uncertainty_values = np.ones((self.uncertainty_grids.shape[0], ))
        self.uncertainty_values = np.random.uniform(
            low=0.95, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))

        self.fig = None
        self.line1 = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, ref_pos, agent_pos_dict=None):
        #self.nu = 1
        self.agent_targets = np.reshape(ref_pos, (self.n_agents, self.n_action))
        self.fail_check = np.zeros(self.n_agents)
        max_distance = 5.0
        min_distance = 0.5
        done = False
        traj_list = []
        drone_crash = False
        reward_list = np.zeros(self.n_agents)

        # self.agent_targets = np.copy(self.agent_pos_goal)

        # for i in range(self.n_agents):
        #     print ("Agent State: ", self.quadrotors[i].state)

        for i in range(self.n_agents):
            pos_list = []

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
                pos0[0], pos0[1], pos0[2], i + 1))
            print("Target X:{0:.3}, Y:{1:.3}, Z:{2:.3} in {3:.3} s.".format(
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
                    done = True
                    reward_list[i] = -1e4
                    break

                current_pos = [self.quadrotors[i].state[0],
                               self.quadrotors[i].state[1], self.quadrotors[i].state[2]]

                reward_list[i] -= 0.025

                if ind % 100 == 0:
                    # if self.visualization:
                    #     self.visualize()
                    agent_pos_dict[i].append([self.quadrotors[i].state[0], self.quadrotors[i].state[1],
                                     self.quadrotors[i].state[2], self.quadrotors[i].state[5]])

                    differences = current_pos - self.uncertainty_grids
                    distances = np.sum(differences * differences, axis=1)
                    indices = distances < self.dist

                    for a in range(self.uncertainty_grids[indices].shape[0]):
                        self.X_list.append(
                            int(self.uncertainty_grids[indices][a, 0]))
                        self.Y_list.append(
                            int(self.uncertainty_grids[indices][a, 1]))
                        self.Z_list.append(
                            int(self.uncertainty_grids[indices][a, 2]))


                    reward_list[i] += 100.0 * np.sum(self.uncertainty_values[indices])
                    # out_of_map = 100*(np.clip(current_pos[0]-self.x_lim, 0, 1e3) +
                    #                   np.clip(current_pos[1]-self.y_lim, 0, 1e3) +
                    #                   np.clip(current_pos[2]-self.z_lim, 0, 1e3))

                    # reward -= out_of_map
                    # min_ind = np.argmin(distances)
                    # if self.uncertainty_values[min_ind] < 0.1:
                    #     neg_reward = np.clip(np.exp(self.grid_visits[min_ind] / 4), 0, 1e3)
                    #     reward -= neg_reward
                    # else:
                    #     reward += 100.0*self.uncertainty_values[min_ind]
                    self.grid_visits[indices] += 1
                    self.uncertainty_values[indices] = np.clip(
                        np.exp(-self.grid_visits[indices]), 1e-6, 1.0)  # Made changes here was 1e-6

                    drone_distances = np.zeros(self.n_agents - 1)
                    for j in range(self.n_agents):
                        if i != j:
                            state_difference = self.quadrotors[i].state - self.quadrotors[j].state
                            drone_distance = np.sqrt(state_difference[0]**2 + state_difference[1]**2 + state_difference[2]**2)
                            if drone_distance < min_distance:
                                reward_list[i] = -1e3
                                # done = True
                            elif drone_distance <= max_distance:
                                reward_list[i] -= 100

                    # print ("current_pos: ", current_pos)
                    # print ("closest grid: ", self.uncertainty_grids[min_ind])
                    
            print("Current X:{0:.3}, Y:{1:.3}, Z:{2:.3}, Reward:{3:.5} \n".format(
                self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2], reward_list[i]))
            

        return self._get_obs(), reward_list, done, agent_pos_dict

    def _get_obs(self):

        for i in range(self.n_agents):
            self.agent_features[i,0] = self.quadrotors[i].state[0] / self.x_lim
            self.agent_features[i,1] = self.quadrotors[i].state[1] / self.y_lim
            self.agent_features[i,2] = self.quadrotors[i].state[2] / self.z_lim

            cnt = 3
            for j in range(self.n_agents):
                if i != j:                    
                    self.agent_features[i,cnt] = (self.quadrotors[i].state[0] - self.quadrotors[j].state[0]) / self.x_lim
                    self.agent_features[i,cnt+1] = (self.quadrotors[i].state[1] - self.quadrotors[j].state[1]) / self.y_lim
                    self.agent_features[i,cnt+2] = (self.quadrotors[i].state[2] - self.quadrotors[j].state[2]) / self.z_lim

                    cnt += 3


        uncertainty_mat = np.reshape(self.uncertainty_values, (1, 1, self.out_shape, self.out_shape))

        return self.agent_features, uncertainty_mat

    def reset(self):
        x = np.zeros((self.n_agents, 2 * self.n_action))
        self.agent_features = np.zeros((self.n_agents, self.n_action + 3*(self.n_agents - 1)))
        self.quadrotors = []
        self.uncertainty_values = uniform(low=0.95, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))
        pos_start = np.zeros((self.n_agents, 3))

        for i in range(0, self.n_agents):
            x_start = uniform(low=-self.x_lim*0.8, high=self.x_lim*0.8)
            y_start = uniform(low=-self.y_lim*0.8, high=self.y_lim*0.8)
            z_start = uniform(low=0.0, high=self.z_lim*0.8)
            pos_start[i,:] = [x_start, y_start, z_start]

            state0 = [x_start, y_start, z_start,
                      0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.quadrotors.append(Quadrotor(state0))

        return self._get_obs(), pos_start

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

    def uncertainty_visualizer(self):
        # prepare some coordinates
        # np.mgrid[-20:20:41j, -20:20:41j, 0:15:16j]
        voxels = np.zeros((41, 41, 16))
        voxels[:, :, :] = False
        # set the colors of each object
        x, y, z = np.indices(np.array(voxels.shape) + 1)

        # and plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # print(self.Z_list)
        for i in range(len(self.X_list)):
            voxels[:, :, self.Z_list[i] - 1][self.X_list[i] +
                                             20, self.Y_list[i] + 20] = True

        ax.voxels(x - 20, y - 20, z, voxels, facecolors='red', edgecolor='k')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 15)
        ax.set_xlabel('X - Dim')
        ax.set_ylabel('Y - Dim')
        ax.set_zlabel('Z - Dim')
        plt.show()

    def visualize(self, pos_list=None, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/drone.png")
            self.drone_transforms = []
            self.drones = []

            for i in range(self.n_agents):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(fname, 2., 2.))
                self.drones[i].add_attr(self.drone_transforms[i])

        for i in range(self.n_agents):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(pos_list[i][0], pos_list[i][1])
            self.drone_transforms[i].set_rotation(pos_list[i][3])

        # N_max = np.max([len(agent_pos_dict[i]) for i in agent_pos_dict.keys()])

        # for i in range(self.n_agents):
        #     self.viewer.add_onetime(self.drones[i])
        #     for j in range(N_max):
        #         if j < len(agent_pos_dict[i]):
        #             pos_angle = agent_pos_dict[i][j]
        #             self.drone_transforms[i].set_translation(pos_angle[0], pos_angle[1])
        #             self.drone_transforms[i].set_rotation(pos_angle[3])        
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
