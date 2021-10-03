import gym
from gym import spaces, error, utils
from gym.utils import seeding
#from gym.envs.classic_control import rendering
import numpy as np
import configparser
from os import path
import itertools
import random
import pdb
from quadrotor_dynamics import Quadrotor
from numpy.random import uniform
from time import sleep
from collections import deque
import warnings



font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):
    def __init__(self, n_agents=1, N_frame=5, visualization=True, is_centralized = True, is_planner = False):
        warnings.filterwarnings('ignore')
        # number of actions per agent which are desired positions and yaw angle
        self.n_action = 6
        self.observation_dim = 4
        self.dim_actions = 1
        self.n_agents = n_agents
        self.visualization = visualization
        self.is_centralized = is_centralized
        self.is_planner = is_planner
        self.action_dict = {0:"Xp", 1:"Xn", 2:"Yp", 3:"Yn", 4:"Zp", 5:"Zn"}

        state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.quadrotors = []
        self.viewer = None
        self.dtau = 1e-3
        self.agent_status = None
        self.agent_is_stuck = None

        if self.is_centralized:
            self.action_space = spaces.Discrete(self.n_action**self.n_agents)
        else:
            self.action_space = spaces.Discrete(self.n_action)

        # intitialize grid information
        self.x_lim = 20  # grid x limit
        self.y_lim = 20  # grid y limit
        self.z_lim = 6  # grid z limit
        self.lim_values = [self.x_lim, self.y_lim, self.z_lim]
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = 82  # width and height for uncertainty matrix
        self.dist = 5.0  # distance threshold
        self.N_closest_grid = 1
        self.neighbour_grids = 8
        self.agent_pos_index = None
        self.safest_indices = None

        # Battery definitions
        self.battery_points = np.array([[-20, -20, 0, -18, -12, 6], [18, 12, 0, 20, 20, 6], 
                                            [-20, 12, 0, -18, 20, 6], [18, -20, 0, 20, -12, 6]])
        self.battery_critical_level = 0.25

        self.battery_positions = None
        self.battery_status = None
        self.battery_indices = None
        self.battery_stack = None

        self.obstacle_start = np.array([[7, 5,0]]) 
        self.obstacle_end = np.array([[9, 18, 6]])
        self.obstacle_points = np.array([[7,5,0,9,18,6]])
        
        self.obstacle_indices = None
        self.obstacle_pos_xy = None

        X, Y, Z = np.mgrid[-self.x_lim : self.x_lim + 0.1 : self.grid_res, 
                           -self.y_lim : self.y_lim + 0.1 : self.grid_res, 
                           0:self.z_lim + 0.1 : 2*self.grid_res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T
        self.uncertainty_values = None
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))

        

        self.N_frame = N_frame # Number of frames to be stacked
        self.frame_update_iter = 2
        self.iteration = None
        self.agents_stacks = [deque([],maxlen=self.N_frame) for _ in range(self.n_agents)]
        self.uncertainty_stacks = deque([],maxlen=self.N_frame)
        self.obstacles_stack = None

        self.action_list = []
        for p in itertools.product([0,1,2,3,4,5], repeat=2):
            self.action_list.append(p)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, iteration, is_centralized):
        self.iteration = iteration
        max_distance = 5.0
        min_distance = 0.5
        uncertainty_constant = 0.0075
        battery_cost = 0.005
        battery_reward = 2.0
        done = False
        reward_list = np.zeros(self.n_agents)
        uncertainty_limit = 0.25
        collision_reward = -10.0
        N_overvisit = 15.0
        obstacle_collision = np.zeros(self.n_agents)
        total_explored_indices = []
        info = dict()

        for i in range(self.n_agents):
            total_explored_indices.append([])

        if is_centralized:
            agents_actions = self.action_list[action]
        else:
            agents_actions = np.reshape(action, (self.n_agents,))
        
        drone_current_pos = np.array([[self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2]] for i in range(self.n_agents)])    
        drone_init_pos = np.copy(drone_current_pos)    

        # print ("\n")

        #increase the map uncertainty
        self.uncertainty_values[self.no_obstacle_indices] = np.clip(
                    self.uncertainty_values[self.no_obstacle_indices] + uncertainty_constant, 1e-6, 1.0)

        # print ("Agent status: ", self.agent_status)

        for agent_ind in range(self.n_agents):
            
            if self.agent_status[agent_ind] == 0:
                # print ("Agent {0} failed, it can't fly any longer!".format(agent_ind+1))
                continue

            current_action = agents_actions[agent_ind]
            drone_prev_state, drone_current_state = self.get_drone_des_grid(agent_ind, current_action)
            current_pos = [drone_current_state[0],drone_current_state[1],drone_current_state[2]]
            self.agent_pos_index[agent_ind] = self.get_closest_grid(current_pos)

            explored_indices = self.get_closest_n_grids(current_pos, self.neighbour_grids)
            current_grid = self.get_closest_grid(current_pos)

            self.battery_status[agent_ind] = np.clip(self.battery_status[agent_ind] - 0*battery_cost, 0.0, 1.0)

            # print ("Agent {0}".format(agent_ind+1))
            # print ("Current action: {0} / {1}".format(agents_actions[agent_ind], self.action_dict[agents_actions[agent_ind]]))
            # print ("Previous state: X:{0:.4}, Y:{1:.4}, Z:{2:.4}".format(drone_prev_state[0], drone_prev_state[1], drone_prev_state[2]))
            # print ("Current state: X:{0:.4}, Y:{1:.4}, Z:{2:.4}".format(drone_current_state[0], drone_current_state[1], drone_current_state[2]))
            
            # if self.check_collision(explored_indices[0:1]): # just check the nearest 1 grid to the drone, whether it collides with the obstacle
            #     # print ("drone_prev_state: ", drone_prev_state)
            #     # print ("drone_current_state: ", drone_current_state)
            #     # print ("Agent {} has collided with the obstacle! It can no longer fly!".format(agent_ind+1))
            #     self.agent_status[agent_ind] = 0 # this agent failed 
            #     obstacle_collision[agent_ind] = 1
            #     reward_list[agent_ind] = collision_reward
            #     self.quadrotors[agent_ind].state = np.copy(drone_prev_state)
            #     self.quadrotors[agent_ind].state[2] = 0.0 # drone falls into (x, y, 0) position. 
            #     # done = True
            #     continue
            
            self.agent_is_stuck[agent_ind] = 0.0
            
            # if self.battery_status[agent_ind] <= 1e-2 and current_grid not in self.battery_indices: # if the battery status of an agent is less than %1, finish the episode
            #     reward_list[agent_ind] = collision_reward
            #     # done = True
            #     self.agent_status[agent_ind] = 0 # this agent failed 
            #     self.quadrotors[agent_ind].state[2] = 0.0 # drone falls into (x, y, 0) position. 
            #     # print ("Agent {} is out of battery! It can no longer fly!".format(agent_ind+1))
            # elif current_grid in self.battery_indices and self.battery_status[agent_ind] <= self.battery_critical_level: #if the agent goes the battery station with low battery, get positive reward
            #     # reward_list[agent_ind] = battery_reward
            #     self.battery_status[agent_ind] = 1.0
            #     # print ("The battery level of Agent {0} is {1:.3}".format(agent_ind+1, self.battery_status[agent_ind]))
            # elif current_grid in self.battery_indices and self.battery_status[agent_ind] > self.battery_critical_level: #if the agent goes the battery station with loaded battery, get negative reward
            #     # reward_list[agent_ind] = -battery_reward
            #     self.battery_status[agent_ind] = 1.0
            #     self.agent_is_stuck[agent_ind] = 1.0
            #     # print ("The battery level of Agent {0} is {1:.3}".format(agent_ind+1, self.battery_status[agent_ind]))

            # if np.sum(self.agent_status) < 1:
            #     print ("No alive agent is left!")
            #     done = True


            total_explored_indices[agent_ind] = explored_indices

            reward_list[agent_ind] += np.sum(self.uncertainty_values[explored_indices]) # max value will be neighbour_grids(=8)


        for agent_ind in range(self.n_agents):
            if obstacle_collision[agent_ind] == 1 or self.agent_status[agent_ind] == 0:
                continue
            else:
                indices = total_explored_indices[agent_ind]
                # exclude the indices of obstacles from the list of visited indices
                to_be_updated_indices = np.setdiff1d(indices, self.obstacle_indices) # obstacle indices are excluded
                to_be_updated_indices = np.setdiff1d(to_be_updated_indices, self.battery_indices) # battery indices are excluded

                self.grid_visits[to_be_updated_indices] += 1
                # self.uncertainty_values[to_be_updated_indices] = np.clip(
                #     np.exp(-self.grid_visits[to_be_updated_indices]/3), 1e-6, 1.0)

                low_uncertainty_indices = np.where(self.uncertainty_values < uncertainty_limit)[0]
            
                # find the visited grids that have low uncertainty values
                overexplored_indices =  np.intersect1d(low_uncertainty_indices, to_be_updated_indices)
                if len(overexplored_indices) > 0:
                    # neg_reward = np.sum(np.clip(np.exp(self.grid_visits[overexplored_indices] / 8), 0, 1))
                    neg_reward = np.sum(np.clip(self.grid_visits[overexplored_indices] / N_overvisit, 0.0, 1.0))
                    reward_list[agent_ind] -= neg_reward

                drone_distances = np.zeros(self.n_agents - 1)
                # for agent_other_ind in range(self.n_agents):
                #     if agent_ind != agent_other_ind:
                #         state_difference = self.quadrotors[agent_ind].state - self.quadrotors[agent_other_ind].state
                        # drone_distance = np.sqrt(state_difference[0]**2 + state_difference[1]**2 + state_difference[2]**2)
                        # if drone_distance < min_distance:
                        #     reward_list[agent_ind] = collision_reward
                        #     reward_list[agent_other_ind] = collision_reward
                        #     # done = True
                        #     self.agent_status[agent_ind] = 0 # this agent failed 
                        #     self.agent_status[agent_other_ind] = 0 # this agent failed 
                        #     self.quadrotors[agent_ind].state[2] = 0.0 # drone falls into (x, y, 0) position. 
                        #     self.quadrotors[agent_other_ind].state[2] = 0.0 # drone falls into (x, y, 0) position. 
                        #     # print ("Agent {} and {} has collided with each other! They can no longer fly!".format(agent_ind+1, agent_other_ind+1))
                        # elif drone_distance <= max_distance:
                        #     reward_list[agent_ind] += (collision_reward/2) 
                        #     reward_list[agent_other_ind] += (collision_reward/2) 
                        
                        

            if self.visualization:
                self.visualize()

        

            if self.iteration % self.frame_update_iter == 0:
                drone_stack = self.get_drone_stack(agent_ind)
                self.agents_stacks[agent_ind].append(drone_stack)


            # print ("Agent {0}".format(agent_ind+1))
            # print ("Current reward: {0:.4}".format(reward_list[agent_ind]))

        # sleep(0.25)

        

        
        # if self.is_centralized:
        #     return self.get_observation(), reward_list.sum(), done, {}
        # else:
        #     return self.get_observation(), reward_list, done, {}

        info['alive_mask'] = np.copy(self.agent_status)

        if self.is_planner:
            return reward_list, done

        return self.get_observation(), reward_list, done, info, [self.quadrotors[i].state for i in range(self.n_agents)]

        

    def get_observation(self):
        uncertainty_map = np.reshape(self.uncertainty_values,(self.out_shape, self.out_shape))
        state_array = np.array([self.quadrotors[i].state[0:3] for i in range(self.n_agents)])
        state_obs = np.zeros((self.n_agents, self.n_agents*3))

        # 1 + n_agents*3 
        #observation list = [battery_status,x11,x12,..x1n,y11,y12,..y1n,z11,z12,..z1n_diff]
        for axis in range(3): #x, y, z axes
            state_tile = np.tile(state_array.T[axis],(self.n_agents,1))
            state_tile_mask = np.copy(state_tile)
            np.fill_diagonal(state_tile_mask, 0)
            state_obs[:,axis*self.n_agents:(axis+1)*self.n_agents] = np.copy(state_tile.T - state_tile_mask) / self.lim_values[axis]

        final_obs = np.c_[self.battery_status, state_obs]

        return final_obs, uncertainty_map


    def get_stack_observation(self):
        # conv_stack(batch,17,84,84) = 5*agent1_pos + 5*agent2_pos + 5*uncertainty_grid + 1*obstacle_grid  + 1*battery_stack

        uncertainty_map = np.reshape(self.uncertainty_values,(self.out_shape, self.out_shape))
        if self.iteration % self.frame_update_iter == 0:
            self.uncertainty_stacks.append(uncertainty_map)

        
        conv_stack = np.zeros((self.N_frame*(self.n_agents+1)+2, self.out_shape, self.out_shape))
        obs_stack = np.zeros((self.n_agents, self.N_frame*(self.n_agents+1)+2, self.out_shape, self.out_shape))
        for agent_ind in range(self.n_agents):
            for frame_ind in range(self.N_frame):
                # agent_ind = 0, 0 1 2 3 4
                # agent_ind = 1, 5 6 7 8 9
                conv_stack[self.N_frame*agent_ind+frame_ind,:,:] = np.copy(self.agents_stacks[agent_ind][frame_ind])

        # uncertainty_stack 10 11 12 13 14
        for frame_ind in range(self.N_frame):
            conv_stack[self.N_frame*(self.n_agents)+frame_ind,:,:] = np.copy(self.uncertainty_stacks[frame_ind])

        conv_stack[-2,:,:] = np.copy(self.obstacles_stack)
        conv_stack[-1,:,:] = np.copy(self.battery_stack)

        for i in range(self.n_agents):
            obs_stack[i,:,:,:] = np.copy(conv_stack)

        return obs_stack, self.battery_status


    def reset(self):
        self.quadrotors = []
        self.uncertainty_values = uniform(low=0.99, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))
        self.agents_stacks = [deque([],maxlen=self.N_frame) for _ in range(self.n_agents)]
        self.uncertainty_stacks = deque([],maxlen=self.N_frame)
        self.agent_status = np.ones(self.n_agents)
        self.agent_pos_index = -1 * np.ones(self.n_agents)
        self.agent_is_stuck = np.zeros(self.n_agents)

        
        self.iteration = 1
        info = dict()
        info['alive_mask'] = np.copy(self.agent_status)

        #There will be two obstacles around (x1,x2,y1,y2)=(-9,-7,5,16) and (x1,x2,y1,y2)=(7,9,-10,10) with -+ 3m deviation in x and y 
        x_rnd = 0 #np.random.uniform(-3,3)
        y_rnd = 0 #np.random.uniform(-3,3)
        # self.obstacle_start = np.array([[-9+x_rnd,5+y_rnd,0],[7+x_rnd, -10+y_rnd,0]]) 
        # self.obstacle_end = np.array([[-7+x_rnd,16+y_rnd,6],[9+x_rnd,10+y_rnd,6]])
        

        self.obstacle_indices = [] #self.get_obstacle_indices()
        
        self.battery_status = np.random.uniform(low=0.95, high=1.0,size=self.n_agents)
        self.battery_status[0] = 0.2

        self.battery_indices = self.get_battery_indices()

        self.uncertainty_values[self.obstacle_indices] = -1.0 # make uncertainty values of obstacle positions -1 so that agents should not get close to them
        self.uncertainty_values[self.battery_indices] = 0.0 # make uncertainty values of battery positions -0 so that agents should not get reward by going there

        self.obstacles_stack = np.zeros(self.uncertainty_grids.shape[0])
        self.obstacles_stack[self.obstacle_indices] = 1
        self.obstacles_stack = np.reshape(self.obstacles_stack,(self.out_shape, self.out_shape))
        self.battery_stack = np.zeros(self.uncertainty_grids.shape[0])
        self.battery_stack[self.battery_indices] = 1
        self.battery_stack = np.reshape(self.battery_stack,(self.out_shape, self.out_shape))

        total_indices = np.arange(self.uncertainty_grids.shape[0])
        self.no_obstacle_indices = np.setdiff1d(total_indices, self.obstacle_indices)


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
        

        uncertainty_map = np.reshape(self.uncertainty_values,(self.out_shape, self.out_shape))
        for j in range(self.N_frame):
            self.uncertainty_stacks.append(uncertainty_map)

        for agent_ind in range(0, self.n_agents):
            total_indices = np.arange(self.uncertainty_grids.shape[0])
            safe_indices = np.setdiff1d(total_indices, self.obstacle_indices)
            self.safest_indices = np.setdiff1d(safe_indices, self.battery_indices)
            closest_grid = np.random.choice(self.safest_indices)
            current_pos = self.uncertainty_grids[closest_grid]

            state0 = [current_pos[0], current_pos[1], current_pos[2],
                      0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.quadrotors.append(Quadrotor(state0))
            self.agent_pos_index[agent_ind] = self.get_closest_grid(current_pos)

            drone_stack = self.get_drone_stack(agent_ind)

            for j in range(self.N_frame):
                self.agents_stacks[agent_ind].append(drone_stack)                


        if not self.is_planner:
            return self.get_observation(), info 


    def get_drone_stack(self, agent_ind):
        drone_closest_grids = self.get_closest_n_grids(self.quadrotors[agent_ind].state[0:3], self.neighbour_grids)
        # print ("drone state: ", self.quadrotors[agent_ind].state[0:3])
        # print ("closest grids: ", self.uncertainty_grids[drone_closest_grids])
        
        drone_stack = np.zeros(self.uncertainty_grids.shape[0])
        drone_stack[drone_closest_grids] = 1
        drone_stack = np.reshape(drone_stack, (self.out_shape, self.out_shape))

        return drone_stack

    def check_collision(self, sorted_drone_indices):
        s = set(self.obstacle_indices)
        for index in sorted_drone_indices:
            if index in s:
                # print ("collided grid: ", index)
                # print ("collided grid position: ", self.uncertainty_grids[index])
                return True

        return False

    def get_neighbor_grids(self, drone_index):
        action_pos_index = dict()

        for i in range(self.n_action):
            agent_pos = np.copy(self.quadrotors[drone_index].state[0:3])
            if i == 0: #action=0, x += 1.0
                agent_pos[0] += self.grid_res
                agent_pos[0] = np.clip(agent_pos[0], -self.x_lim,  self.x_lim)
            elif i == 1: #action=1, x -= 1.0
                agent_pos[0] -= self.grid_res
                agent_pos[0] = np.clip(agent_pos[0], -self.x_lim,  self.x_lim)
            elif i == 2: #action=2, y += 1.0
                agent_pos[1] += self.grid_res
                agent_pos[1] = np.clip(agent_pos[1], -self.y_lim,  self.y_lim)
            elif i == 3: #action=3, y -= 1.0
                agent_pos[1] -= self.grid_res
                agent_pos[1] = np.clip(agent_pos[1], -self.y_lim,  self.y_lim)
            elif i == 4: #action=4, z += 2.0
                agent_pos[2] += 2*self.grid_res
                agent_pos[2] = np.clip(agent_pos[2], 0.0,  self.z_lim)
            elif i == 5: #action=5, z += 2.0
                agent_pos[2] -= 2*self.grid_res
                agent_pos[2] = np.clip(agent_pos[2], 0.0,  self.z_lim)

            action_pos_index[i] = self.get_closest_grid(agent_pos)

        return action_pos_index

    def get_drone_des_grid(self, drone_index, discrete_action):
        drone_prev_state = np.copy(self.quadrotors[drone_index].state)

        if discrete_action == 0: #action=0, x += 1.0
            self.quadrotors[drone_index].state[0] += self.grid_res
            self.quadrotors[drone_index].state[0] = np.clip(self.quadrotors[drone_index].state[0], -self.x_lim,  self.x_lim)
        elif discrete_action == 1: #action=1, x -= 1.0
            self.quadrotors[drone_index].state[0] -= self.grid_res
            self.quadrotors[drone_index].state[0] = np.clip(self.quadrotors[drone_index].state[0], -self.x_lim,  self.x_lim)
        elif discrete_action == 2: #action=2, y += 1.0
            self.quadrotors[drone_index].state[1] += self.grid_res
            self.quadrotors[drone_index].state[1] = np.clip(self.quadrotors[drone_index].state[1], -self.y_lim,  self.y_lim)
        elif discrete_action == 3: #action=3, y -= 1.0
            self.quadrotors[drone_index].state[1] -= self.grid_res
            self.quadrotors[drone_index].state[1] = np.clip(self.quadrotors[drone_index].state[1], -self.y_lim,  self.y_lim)
        elif discrete_action == 4: #action=4, z += 2.0
            self.quadrotors[drone_index].state[2] += self.grid_res*2
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], 0.0,  self.z_lim)
        elif discrete_action == 5: #action=5, z += 2.0
            self.quadrotors[drone_index].state[2] -= self.grid_res*2
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], 0.0,  self.z_lim)
        elif discrete_action == -1: #action=-1 stop
            print ("No action executed!")
        else:
            print ("Invalid discrete action!")

        drone_current_state = np.copy(self.quadrotors[drone_index].state)
        return drone_prev_state, drone_current_state


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


    def get_obstacle_indices_2(self):
        obstacle_indices = []
        obstacle_pos = []
        obstacle_indices_unsquezed = []

        for i in range(self.obstacle_start.shape[0]):
            x_range = np.arange(-self.grid_res/2+self.obstacle_start[i,0], self.obstacle_end[i,0]+self.grid_res/2, self.grid_res/4)
            y_range = np.arange(-self.grid_res/2+self.obstacle_start[i,1], self.obstacle_end[i,1]+self.grid_res/2, self.grid_res/4)
            z_range = np.arange(-self.grid_res/2+self.obstacle_start[i,2], self.obstacle_end[i,2]+self.grid_res/2, self.grid_res/2)

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

    def get_obstacle_indices(self):
        obstacle_indices = []
        lst = []
        for location in self.obstacle_points:
            xyz = np.mgrid[location[0]:location[3]+0.1:self.grid_res,
                            location[1]:location[4]+0.1:self.grid_res,
                            location[2]:location[5]+0.1:2*self.grid_res].reshape(3,-1).T
            lst.append(xyz)
            
        self.obstacle_positions = np.vstack((lst[i] for i in range(len(lst))))
        array_of_tuples = map(tuple, self.obstacle_positions)
        self.obstacle_positions = tuple(array_of_tuples)

        for pos in self.obstacle_positions:
            current_pos = np.array([pos[0], pos[1], pos[2]])
            current_ind = self.get_closest_grid(current_pos)
            obstacle_indices.append(current_ind)

        # obst_x_min = np.min(self.uncertainty_grids[obstacle_indices,0], axis=0)
        # obst_x_max = np.max(self.uncertainty_grids[obstacle_indices,0], axis=0)
        # obst_y_min = np.min(self.uncertainty_grids[obstacle_indices,1], axis=0)
        # obst_y_max = np.max(self.uncertainty_grids[obstacle_indices,1], axis=0)
        # obst_z_min = np.min(self.uncertainty_grids[obstacle_indices,2], axis=0)
        # obst_z_max = np.max(self.uncertainty_grids[obstacle_indices,2], axis=0)

        # print ("obst_x_min: ", obst_x_min)
        # print ("obst_x_max: ", obst_x_max)
        # print ("obst_y_min: ", obst_y_min)
        # print ("obst_y_max: ", obst_y_max)
        # print ("obst_z_min: ", obst_z_min)
        # print ("obst_z_max: ", obst_z_max)


        return obstacle_indices

    def get_battery_indices(self):
        battery_indices = []
        lst = []
        for location in self.battery_points:
            xyz = np.mgrid[location[0]:location[3]+0.1:self.grid_res,
                            location[1]:location[4]+0.1:self.grid_res,
                            location[2]:location[5]+0.1:2*self.grid_res].reshape(3,-1).T
            lst.append(xyz)
            
        self.battery_positions = np.vstack((lst[0],lst[1], lst[2], lst[3]))
        array_of_tuples = map(tuple, self.battery_positions)
        self.battery_positions = tuple(array_of_tuples)

        for pos in self.battery_positions:
            current_pos = np.array([pos[0], pos[1], pos[2]])
            current_ind = self.get_closest_grid(current_pos)
            battery_indices.append(current_ind)

        return battery_indices


    def visualize(self, agent_pos_dict=None, mode='human'):
        charge_station = []
        station_transform = []

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/drone.png")

            # obstacle_pos_xy = [x_min, y_min, z_min, x_max, y_max, z_max]
            # for i in range(len(self.obstacle_points)):
            #     obstacle = rendering.make_polygon([(self.obstacle_points[i][0],self.obstacle_points[i][1]), 
            #                                     (self.obstacle_points[i][0],self.obstacle_points[i][4]), 
            #                                     (self.obstacle_points[i][3],self.obstacle_points[i][4]), 
            #                                     (self.obstacle_points[i][3],self.obstacle_points[i][1])])

            #     obstacle_transform = rendering.Transform()
            #     obstacle.add_attr(obstacle_transform)
            #     obstacle.set_color(.8, .3, .3)
            #     self.viewer.add_geom(obstacle)

            # obstacle_pos_xy = [x_min, y_min, z_min, x_max, y_max, z_max]
            # l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            # axleoffset =cartheight/4.0
            # cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            for j in range(self.battery_points.shape[0]):
                charge_station.append(rendering.make_polygon([(self.battery_points[j][0],self.battery_points[j][1]), 
                                                (self.battery_points[j][0],self.battery_points[j][4]), 
                                                (self.battery_points[j][3],self.battery_points[j][4]), 
                                                (self.battery_points[j][3],self.battery_points[j][1])]))

                station_transform.append(rendering.Transform())
                charge_station[j].add_attr(station_transform[j])
                charge_station[j].set_color(.1, .5, .8)
                self.viewer.add_geom(charge_station[j])

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


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
