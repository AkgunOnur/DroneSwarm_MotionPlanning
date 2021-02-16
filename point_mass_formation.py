import gym
from gym import spaces, error, utils
from gym.utils import seeding
# from gym.envs.classic_control import rendering
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



font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=1, visualization=True):

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

        self.v_average = 1.0
        self.period_denum = 1.0
        self.dtau = 1e-3

        self.xd_dot, self.yd_dot, self.zd_dot = 0, 0, 0
        self.xd_dotdot, self.yd_dotdot, self.zd_dotdot = 0, 0, 0

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
        self.N_closest_grid = 4

        X, Y, Z = np.mgrid[-self.x_lim : self.x_lim + 0.1 : self.res, 
                           -self.y_lim : self.y_lim + 0.1 : self.res, 
                           0:self.z_lim + 0.1 : 2*self.res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T
        self.uncertainty_values = None
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))

        self.obstacle_start = None
        self.obstacle_end = None
        self.obstacle_indices = None
        self.obstacle_pos_xy = None

        self.fig = None
        self.line1 = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, ref_pos):
        #self.nu = 1
        self.agent_targets = np.reshape(ref_pos, (self.n_agents, self.n_action))
        self.fail_check = np.zeros(self.n_agents)
        max_distance = 5.0
        min_distance = 0.5
        alpha = 0.1
        done = False
        traj_list = []
        drone_crash = False
        reward_list = np.zeros(self.n_agents)

        agent_pos_dict = {}
        
        drone_current_pos = np.array([[self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2]] for i in range(self.n_agents)])    
        drone_init_pos = np.copy(drone_current_pos)    
        drone_prev_pos = np.copy(drone_current_pos)

        reached_target = np.zeros(self.n_agents)

        N = int(np.sum(np.abs(np.max(self.agent_targets - drone_current_pos, axis=0))) * 10)
        N = 50


        for k in range(N):
            u = (self.agent_targets - drone_current_pos)*100
            drone_current_pos = drone_current_pos + (u + 0*np.random.uniform(-1,1)) * self.dtau 

            reward_list[reached_target==0] -= 10 # Give moving agents negative reward

            if (k+1) % 5 == 0:
                diff_pos = drone_current_pos - drone_prev_pos
                drone_prev_pos = np.copy(drone_current_pos)
                
                for i in range(self.n_agents):
                    current_pos = [drone_current_pos[i,0],drone_current_pos[i,1],drone_current_pos[i,2]]
                    dist_to_target = drone_current_pos[i,:] - self.agent_targets[i,:]

                    if reached_target[i]: # If this agent reaches the target, then pass the next one
                        continue

                    differences = current_pos - self.uncertainty_grids
                    distances = np.sum(differences * differences, axis=1)
                    indices = distances < self.dist
                    sorted_drone_indices = sorted(range(len(distances)), key=lambda k: distances[k])

                    if self.check_collision(sorted_drone_indices[0:self.N_closest_grid // 2]): # just check the nearest 2 grids to the drone, whether it collides with the obstacle
                        print ("Agent {} has collided with the obstacle!".format(i+1))
                        reward_list[i] = -1e4
                        done = True
                        break

                    reward_list[i] += 100.0 * np.sum(self.uncertainty_values[indices])
                    # out_of_map = 100*(np.clip(current_pos[0]-self.x_lim, 0, 1e3) +
                    #                   np.clip(current_pos[1]-self.y_lim, 0, 1e3) +
                    #                   np.clip(current_pos[2]-self.z_lim, 0, 1e3))

                    # reward -= out_of_map
                    self.grid_visits[indices] += 1
                    self.uncertainty_values[indices] = np.clip(
                        np.exp(-self.grid_visits[indices]/3), -1.0, 1.0)
                
                    overexplored_indices = np.array(self.uncertainty_values < 0.1) & np.array(indices)
                    if np.sum(overexplored_indices) > 0:
                        neg_reward = np.sum(np.clip(np.exp(self.grid_visits[overexplored_indices] / 5), 0, 1e2))
                        reward_list[i] -= neg_reward

                    drone_distances = np.zeros(self.n_agents - 1)
                    for j in range(self.n_agents):
                        if i != j:
                            state_difference = self.quadrotors[i].state - self.quadrotors[j].state
                            drone_distance = np.sqrt(state_difference[0]**2 + state_difference[1]**2 + state_difference[2]**2)
                            if drone_distance < min_distance:
                                reward_list[i] = -1e4
                                reward_list[j] = -1e4
                                print ("Agent {} and {} has collided with each other!".format(i+1, j+1))
                                done = True
                            elif drone_distance <= max_distance:
                                reward_list[i] -= 100

                    
                    if np.sum(np.abs(dist_to_target)) < 0.1:
                        reached_target[i] = 1

                if self.visualization:
                    self.visualize()


            if np.sum(reached_target) == self.n_agents or done: # If all agents reaches their targets or any of them fails to do it
                break 
        

        for i in range(self.n_agents):
            self.quadrotors[i].state[0:3] = [drone_current_pos[i,0], drone_current_pos[i,1], drone_current_pos[i,2]]
            print("Initial X:{0:.3}, Y:{1:.3}, Z:{2:.3} of Agent {3}".format(
                drone_init_pos[i,0], drone_init_pos[i,1], drone_init_pos[i,2], i+1))
            print("Target X:{0:.3}, Y:{1:.3}, Z:{2:.3}".format(
                self.agent_targets[i,0], self.agent_targets[i,1], self.agent_targets[i,2]))
            print("Final X:{0:.3}, Y:{1:.3}, Z:{2:.3}, Reward:{3:.5} \n".format(
                self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2], reward_list[i]))
            
        return self._get_obs(), reward_list, done, {}

    def _get_obs(self):
        conv_stack = np.zeros((self.out_shape, self.out_shape, self.n_agents + 2)) # it will be in dimension 82x82x4 
        # In the first 2 stacks, there will be position of agents (closest 4 grids to the agent will be 1, others will be 0)
        # In the third stack, there will be uncertainty matrix, whose elements are between 0 and 1
        # In the fourth stack, there will be positions of obstacles (positions of obstacles are 1)

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

            drone_closest_grids = self.get_closest_n_grids(self.quadrotors[i].state[0:3], self.N_closest_grid)
            drone_stack = np.zeros(self.uncertainty_grids.shape[0])
            drone_stack[drone_closest_grids] = 1
            drone_stack = np.reshape(drone_stack, (self.out_shape, self.out_shape))

            conv_stack[:,:,i] = np.copy(drone_stack)

        uncertainty_stack = np.reshape(self.uncertainty_values,(self.out_shape, self.out_shape))
        conv_stack[:,:,self.n_agents] = np.copy(uncertainty_stack)

        obstacles_stack = np.zeros(self.uncertainty_grids.shape[0])
        obstacles_stack[self.obstacle_indices] = 1
        obstacles_stack = np.reshape(obstacles_stack,(self.out_shape, self.out_shape))
        conv_stack[:,:,self.n_agents+1] = np.copy(obstacles_stack)

        conv_stack = np.reshape(conv_stack, (1, self.n_agents+2, self.out_shape, self.out_shape))

        return self.agent_features, conv_stack

    def reset(self):
        x = np.zeros((self.n_agents, 2 * self.n_action))
        self.agent_features = np.zeros((self.n_agents, self.n_action + 3*(self.n_agents - 1)))
        self.quadrotors = []
        self.uncertainty_values = uniform(low=0.99, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))
        pos_start = np.zeros((self.n_agents, 3))
        
        self.viewer = None
        

        #There will be two obstacles around (x1,x2,y1,y2)=(-9,-7,5,16) and (x1,x2,y1,y2)=(7,9,-10,10) with -+ 3m deviation in x and y 
        x_rnd = np.random.uniform(-3,3)
        y_rnd = np.random.uniform(-3,3)
        self.obstacle_start = np.array([[-9+x_rnd,5+y_rnd,0],[7+x_rnd, -10+y_rnd,0]]) 
        self.obstacle_end = np.array([[-7+x_rnd,16+y_rnd,6],[9+x_rnd,10+y_rnd,6]])

        self.obstacle_indices, self.obstacle_pos_xy = self.get_obstacle_indices()

        self.uncertainty_values[self.obstacle_indices] = -1.0 # make uncertainty values of obstacle positions -1 so that agents should not get close to them


        # Debugging to check collision

        # print ("obstacle_indices: ", self.obstacle_indices)
        # print ("obstacle_pos_xy: ", self.obstacle_pos_xy)
        
        # state0 = [-9.5, 5.5, 4.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # self.quadrotors.append(Quadrotor(state0))
        # state0 = [7.0, 8.5, 4.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # self.quadrotors.append(Quadrotor(state0))

        # drone_current_pos = np.array([[self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2]] for i in range(self.n_agents)])

        # for i in range(self.n_agents):
        #     current_pos = [drone_current_pos[i,0],drone_current_pos[i,1],drone_current_pos[i,2]]

        #     differences = current_pos - self.uncertainty_grids
        #     distances = np.sum(differences * differences, axis=1)
        #     sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        #     print ("\n Agent: ", (i+1))
        #     print ("drone_position: ", self.quadrotors[i].state[0:4])
        #     print ("drone_grid_indices: ", sorted_indices[0:4])
        #     print ("drone_grid_positions: ", self.uncertainty_grids[sorted_indices[0:4]])
            
            
        #     self.check_collision(sorted_indices[0:4])
        # stop        
        

        for i in range(0, self.n_agents):
            x_start = uniform(low=-self.x_lim, high=self.x_lim)
            y_start = uniform(low=-self.y_lim, high=self.y_lim)
            z_start = uniform(low=0.0, high=self.z_lim)
            pos_start[i,:] = [x_start, y_start, z_start]

            state0 = [x_start, y_start, z_start,
                      0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.quadrotors.append(Quadrotor(state0))

        return self._get_obs(), pos_start

    def check_collision(self, sorted_drone_indices):
        s = set(self.obstacle_indices)
        for index in sorted_drone_indices:
            if index in s:
                # print ("collided grid: ", index)
                # print ("collided grid position: ", self.uncertainty_grids[index])
                return True

        return False


    def get_closest_n_grids(self, current_pos, n):
        differences = current_pos-self.uncertainty_grids
        distances = np.sum(differences*differences,axis=1)
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        
        return sorted_indices[0:n]

    def get_closest_grid(self, current_pos):
        differences = current_pos-self.uncertainty_grids
        distances = np.sum(differences*differences,axis=1)
        min_ind = np.argmin(distances)
        
        return min_ind


    def get_obstacle_indices(self):
        obstacle_indices = []
        obstacle_pos = []
        obstacle_indices_unsquezed = []

        for i in range(self.obstacle_start.shape[0]):
            x_range = np.arange(-self.res/2+self.obstacle_start[i,0], self.obstacle_end[i,0]+self.res/2, self.res/4)
            y_range = np.arange(-self.res/2+self.obstacle_start[i,1], self.obstacle_end[i,1]+self.res/2, self.res/4)
            z_range = np.arange(-self.res/2+self.obstacle_start[i,2], self.obstacle_end[i,2]+self.res/2, self.res/2)

            indices = []
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        current_pos = np.array([x,y,z])
                        current_ind = self.get_closest_grid(current_pos)
                        if current_ind not in indices:
                            indices.append(current_ind)
            
            obst_x_min = np.min(self.uncertainty_grids[indices,0], axis=0)
            obst_x_max = np.max(self.uncertainty_grids[indices,0], axis=0)
            obst_y_min = np.min(self.uncertainty_grids[indices,1], axis=0)
            obst_y_max = np.max(self.uncertainty_grids[indices,1], axis=0)

            obstacle_pos.append([obst_x_min, obst_x_max, obst_y_min, obst_y_max])
            obstacle_indices.append(indices)

        
        for i in range(len(obstacle_indices)):
            for j in range(len(obstacle_indices[0])):
                obstacle_indices_unsquezed.append(obstacle_indices[i][j])
        
        return obstacle_indices_unsquezed, obstacle_pos


    def visualize(self, agent_pos_dict=None, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/drone.png")

            # obstacle_pos_xy = [x_min, x_max, y_min, y_max]
            for i in range(len(self.obstacle_pos_xy)):
                obstacle = rendering.make_polygon([(self.obstacle_pos_xy[i][0],self.obstacle_pos_xy[i][2]), 
                                                (self.obstacle_pos_xy[i][0],self.obstacle_pos_xy[i][3]), 
                                                (self.obstacle_pos_xy[i][1],self.obstacle_pos_xy[i][3]), 
                                                (self.obstacle_pos_xy[i][1],self.obstacle_pos_xy[i][2])])

                obstacle_transform = rendering.Transform()
                obstacle.add_attr(obstacle_transform)
                obstacle.set_color(.8, .3, .3)
                self.viewer.add_geom(obstacle)

            self.drone_transforms = []
            self.drones = []

            for i in range(self.n_agents):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(fname, 2., 2.))
                self.drones[i].add_attr(self.drone_transforms[i])

        
        
        for i in range(self.n_agents):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(self.quadrotors[i].state[0], self.quadrotors[i].state[1])
            self.drone_transforms[i].set_rotation(self.quadrotors[i].state[5])

        # N_max = np.max([len(agent_pos_dict[i]) for i in agent_pos_dict.keys()])

        # for i in range(self.n_agents):
        #     self.viewer.add_onetime(self.drones[i])
        #     for j in range(N_max):
        #         if j < len(agent_pos_dict[i]):
        #             pos_angle = agent_pos_dict[i][j]
        #             self.drone_transforms[i].set_translation(pos_angle[0], pos_angle[1])
        #             self.drone_transforms[i].set_rotation(pos_angle[3])
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
