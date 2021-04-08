import gym
from gym import spaces, error, utils
from gym.utils import seeding
# from gym.envs.classic_control import rendering
import numpy as np
import configparser
from os import path
import itertools
import random
import pdb
from quadrotor_dynamics import Quadrotor
from numpy.random import uniform
from trajectory import Trajectory
from time import sleep
from collections import deque



font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=1, N_frame=5, visualization=True, is_centralized = True):

        # number of actions per agent which are desired positions and yaw angle
        self.n_action = 6
        self.n_agents = n_agents
        self.visualization = visualization
        self.is_centralized = is_centralized
        self.action_dict = {0:"Xp", 1:"Xn", 2:"Yp", 3:"Yn", 4:"Zp", 5:"Zn"}

        state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.quadrotors = []
        self.viewer = None
        self.dtau = 1e-3

        if self.is_centralized:
            self.action_space = spaces.Discrete(self.n_action**self.n_agents)
        else:
            self.action_space = spaces.Discrete(self.n_action)

        # intitialize grid information
        self.x_lim = 20  # grid x limit
        self.y_lim = 20  # grid y limit
        self.z_lim = 6  # grid z limit
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = 82  # width and height for uncertainty matrix
        self.dist = 5.0  # distance threshold
        self.N_closest_grid = 4
        self.neighbour_grids = 8
        

        X, Y, Z = np.mgrid[-self.x_lim : self.x_lim + 0.1 : self.grid_res, 
                           -self.y_lim : self.y_lim + 0.1 : self.grid_res, 
                           0:self.z_lim + 0.1 : 2*self.grid_res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T
        self.uncertainty_values = None
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))

        self.obstacle_start = None
        self.obstacle_end = None
        self.obstacle_indices = None
        self.obstacle_pos_xy = None

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
        done = False
        reward_list = np.zeros(self.n_agents)
        uncertainty_limit = 0.25
        collision_reward = -10.0
        N_overvisit = 15.0
        obstacle_collision = np.zeros(self.n_agents)
        total_explored_indices = []
        for i in range(self.n_agents):
            total_explored_indices.append([])

        if is_centralized:
            agents_actions = self.action_list[action]
        else:
            agents_actions = np.reshape(action, (self.n_agents,))
        
        drone_current_pos = np.array([[self.quadrotors[i].state[0], self.quadrotors[i].state[1], self.quadrotors[i].state[2]] for i in range(self.n_agents)])    
        drone_init_pos = np.copy(drone_current_pos)    

        # print ("\n")
        for agent_ind in range(self.n_agents):
            current_action = agents_actions[agent_ind]
            drone_prev_state, drone_current_state = self.get_drone_des_grid(agent_ind, current_action)
            current_pos = [drone_current_state[0],drone_current_state[1],drone_current_state[2]]
            explored_indices = self.get_closest_n_grids(current_pos, self.neighbour_grids)

            # print ("Agent {0}".format(agent_ind+1))
            # print ("Current action: {0} / {1}".format(agents_actions[agent_ind], self.action_dict[agents_actions[agent_ind]]))
            # print ("Previous state: X:{0:.4}, Y:{1:.4}, Z:{2:.4}".format(drone_prev_state[0], drone_prev_state[1], drone_prev_state[2]))
            # print ("Current state: X:{0:.4}, Y:{1:.4}, Z:{2:.4}".format(drone_current_state[0], drone_current_state[1], drone_current_state[2]))
            
            if self.check_collision(explored_indices[0:self.N_closest_grid]): # just check the nearest 4 grids to the drone, whether it collides with the obstacle
                # print ("drone_prev_state: ", drone_prev_state)
                # print ("drone_current_state: ", drone_current_state)
                print ("Agent {} has collided with the obstacle!".format(agent_ind+1))
                obstacle_collision[agent_ind] = 1
                reward_list[agent_ind] = collision_reward
                self.quadrotors[agent_ind].state = np.copy(drone_prev_state)
                done = True
                continue

            total_explored_indices[agent_ind] = explored_indices

            reward_list[agent_ind] += np.sum(self.uncertainty_values[explored_indices]) # max value will be neighbour_grids(=14)


        for agent_ind in range(self.n_agents):
            if obstacle_collision[agent_ind] == 1:
                continue
            else:
                indices = total_explored_indices[agent_ind]
                # exclude the indices of obstacles from the list of visited indices
                to_be_updated_indices = np.setdiff1d(indices, self.obstacle_indices)

                self.grid_visits[to_be_updated_indices] += 1
                self.uncertainty_values[to_be_updated_indices] = np.clip(
                    np.exp(-self.grid_visits[to_be_updated_indices]/3), 1e-6, 1.0)

                low_uncertainty_indices = np.where(self.uncertainty_values < uncertainty_limit)[0]
            
                # find the visited grids that have low uncertainty values
                overexplored_indices =  np.intersect1d(low_uncertainty_indices, to_be_updated_indices)
                if len(overexplored_indices) > 0:
                    # neg_reward = np.sum(np.clip(np.exp(self.grid_visits[overexplored_indices] / 8), 0, 1))
                    neg_reward = np.sum(np.clip(self.grid_visits[overexplored_indices] / N_overvisit, 0.0, 1.0))
                    reward_list[agent_ind] -= neg_reward

                drone_distances = np.zeros(self.n_agents - 1)
                for agent_other_ind in range(self.n_agents):
                    if agent_ind != agent_other_ind:
                        state_difference = self.quadrotors[agent_ind].state - self.quadrotors[agent_other_ind].state
                        drone_distance = np.sqrt(state_difference[0]**2 + state_difference[1]**2 + state_difference[2]**2)
                        if drone_distance < min_distance:
                            reward_list[agent_ind] = collision_reward
                            reward_list[agent_other_ind] = collision_reward
                            done = True
                            print ("Agent {} and {} has collided with each other!".format(agent_ind+1, agent_other_ind+1))
                        elif drone_distance <= max_distance:
                            reward_list[agent_ind] += (collision_reward/2) 
                            reward_list[agent_other_ind] += (collision_reward/2) 
                        
                        

            if self.visualization:
                self.visualize()

        

            if self.iteration % self.frame_update_iter == 0:
                drone_stack = self.get_drone_stack(agent_ind)
                self.agents_stacks[agent_ind].append(drone_stack)


            # print ("Agent {0}".format(agent_ind+1))
            # print ("Current reward: {0:.4}".format(reward_list[agent_ind]))

            # sleep(2.0)

        
        if self.is_centralized:
            return self.get_observation(), reward_list.sum(), done, {}
        else:
            return self.get_observation(), reward_list, done, {}

        

    def get_observation(self):
        # conv_stack(batch,4,84,84) input_channel = 4
        # conv_stack(batch,16,84,84) = 5*agent1_pos + 5*agent2_pos + 5*uncertainty_grid + 1*obstacle_grid  
        # conv_stack(batch,5,4,84,84)

        # In the first 2 stacks, there will be position of agents (closest 4 grids to the agent will be 1, others will be 0)
        # In the third stack, there will be uncertainty matrix, whose elements are between 0 and 1
        # In the fourth stack, there will be positions of obstacles (positions of obstacles are 1)

        uncertainty_map = np.reshape(self.uncertainty_values,(self.out_shape, self.out_shape))
        if self.iteration % self.frame_update_iter == 0:
            self.uncertainty_stacks.append(uncertainty_map)

        
        conv_stack = np.zeros((self.N_frame*(self.n_agents+1)+1, self.out_shape, self.out_shape))
        for agent_ind in range(self.n_agents):
            for frame_ind in range(self.N_frame):
                # agent_ind = 0, 0 1 2 3 4
                # agent_ind = 1, 5 6 7 8 9
                conv_stack[self.N_frame*agent_ind+frame_ind,:,:] = np.copy(self.agents_stacks[agent_ind][frame_ind])

        # uncertainty_stack 10 11 12 13 14
        for frame_ind in range(self.N_frame):
            conv_stack[self.N_frame*(self.n_agents)+frame_ind,:,:] = np.copy(self.uncertainty_stacks[frame_ind])

        conv_stack[-1,:,:] = np.copy(self.obstacles_stack)

        return conv_stack


    def reset(self):
        self.quadrotors = []
        self.uncertainty_values = uniform(low=0.99, high=1.0, size=(self.uncertainty_grids.shape[0],))
        self.grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))
        self.agents_stacks = [deque([],maxlen=self.N_frame) for _ in range(self.n_agents)]
        self.uncertainty_stacks = deque([],maxlen=self.N_frame)
        
        self.iteration = 1

        #There will be two obstacles around (x1,x2,y1,y2)=(-9,-7,5,16) and (x1,x2,y1,y2)=(7,9,-10,10) with -+ 3m deviation in x and y 
        x_rnd = 0 #np.random.uniform(-3,3)
        y_rnd = 0 #np.random.uniform(-3,3)
        # self.obstacle_start = np.array([[-9+x_rnd,5+y_rnd,0],[7+x_rnd, -10+y_rnd,0]]) 
        # self.obstacle_end = np.array([[-7+x_rnd,16+y_rnd,6],[9+x_rnd,10+y_rnd,6]])
        self.obstacle_start = np.array([[7+x_rnd, 5+y_rnd,0]]) 
        self.obstacle_end = np.array([[9+x_rnd,20+y_rnd,6]])

        self.obstacle_indices, self.obstacle_pos_xy = self.get_obstacle_indices()

        self.uncertainty_values[self.obstacle_indices] = -1.0 # make uncertainty values of obstacle positions -1 so that agents should not get close to them

        self.obstacles_stack = np.zeros(self.uncertainty_grids.shape[0])
        self.obstacles_stack[self.obstacle_indices] = 1
        self.obstacles_stack = np.reshape(self.obstacles_stack,(self.out_shape, self.out_shape))


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
            closest_grid = np.random.choice(safe_indices)
            current_pos = self.uncertainty_grids[closest_grid]

            state0 = [current_pos[0], current_pos[1], current_pos[2],
                      0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.quadrotors.append(Quadrotor(state0))
            drone_stack = self.get_drone_stack(agent_ind)

            for j in range(self.N_frame):
                self.agents_stacks[agent_ind].append(drone_stack)                


        return self.get_observation()


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
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], 0.5,  self.z_lim)
        elif discrete_action == 5: #action=5, z += 2.0
            self.quadrotors[drone_index].state[2] -= self.grid_res*2
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], 0.5,  self.z_lim)
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


    def get_obstacle_indices(self):
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
