import gym
from gym import spaces, error, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import configparser
from os import path
import itertools
import random
import pdb
from quadrotor_dynamics import Quadrotor, Drone, Bot
from numpy.random import uniform
from time import sleep
from collections import deque



font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=2, n_bots = 2, N_frame=5, visualization=False, is_centralized = False):

        # number of actions per agent which are desired positions and yaw angle
        self.seed()
        self.n_action = 6
        self.observation_dim = 4
        self.dim_actions = 1
        self.n_agents = n_agents
        self.n_bots = n_bots
        self.visualization = visualization
        self.is_centralized = is_centralized
        self.action_dict = {0:"Xp", 1:"Xn", 2:"Yp", 3:"Yn"}

        #state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.quadrotors = []
        self.viewer = None
        self.dtau = 1e-3

        if self.is_centralized:
            self.action_space = spaces.Discrete(self.n_action**self.n_agents)
        else:
            self.action_space = spaces.Discrete(self.n_action)

        # intitialize grid information
        self.x_lim = 41  # grid x limit
        self.y_lim = 41  # grid y limit
        self.z_lim = 12
        self.lim_values = [self.x_lim, self.y_lim, self.z_lim]
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = 82  # width and height for uncertainty matrix
        self.N_closest_grid = 4
        self.neighbour_grids = 8
        

        X, Y= np.mgrid[-self.x_lim : self.x_lim : self.grid_res, 
                           -self.y_lim : self.y_lim : self.grid_res]

        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten())).T

        print(self.uncertainty_grids.shape)

        self.N_frame = N_frame # Number of frames to be stacked
        self.frame_update_iter = 2
        self.iteration = None
        self.agents_stacks = [deque([],maxlen=self.N_frame) for _ in range(self.n_agents)]
        self.bots_stacks = [deque([],maxlen=self.N_frame) for _ in range(self.n_bots)]

        self.action_list = []
        for p in itertools.product([0,1,2,3], repeat=2):
            self.action_list.append(p)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, iteration, is_centralized):
        self.iteration = iteration
        done = False
        reward_list = np.ones(self.n_agents) * (-2)

        if is_centralized:
            agents_actions = self.action_list[action]
        else:
            agents_actions = np.reshape(action[0], (self.n_agents,))
        
        #drone_current_pos = np.array([[self.quadrotors[i].state[0], self.quadrotors[i].state[1]] for i in range(self.n_agents)])      

        for agent_ind in range(self.n_agents):
            current_action = agents_actions[agent_ind]
            drone_current_state = self.get_drone_des_grid(agent_ind, current_action)
            #current_pos = [drone_current_state[0],drone_current_state[1]]


        #if self.iteration % self.frame_update_iter == 0:
        #    for agent_ind in range(self.n_agents):
        #        drone_stack = self.get_drone_stack(agent_ind)
        #        self.agents_stacks[agent_ind].append(drone_stack)

        #    for bot_ind in range(self.n_bots):
        #        bot_stack = self.get_bot_stack(bot_ind)
        #        self.bots_stacks[bot_ind].append(bot_stack)


        for agent_ind in range(self.n_agents):
            for other_agents_ind in range(self.n_agents):
                if agent_ind != other_agents_ind:
                    #collision_state_difference = self.quadrotors[agent_ind].state - self.quadrotors[other_agents_ind].state
                    #collision_distance = np.sqrt(collision_state_difference[0]**2 + collision_state_difference[1]**2 + collision_state_difference[2]**2)
                    collision_distance = np.linalg.norm(self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (collision_distance <= 7) and self.quadrotors[agent_ind].is_alive and self.quadrotors[other_agents_ind].is_alive:
                        done = True
                        self.quadrotors[agent_ind].is_alive = False
                        self.quadrotors[other_agents_ind].is_alive = False
                        self.quadrotors[agent_ind].state[2] = 0.0
                        self.quadrotors[other_agents_ind].state[2] = 0.0
                        reward_list[agent_ind] -= 300
                        reward_list[other_agents_ind] -= 300
            
            if not done:
                for bot_ind in range(self.n_bots):
                    #state_difference = self.quadrotors[agent_ind].state - self.bots[bot_ind].state
                    #drone_distance = np.sqrt(state_difference[0]**2 + state_difference[1]**2 + state_difference[2]**2)
                    drone_distance = np.linalg.norm(self.quadrotors[agent_ind].state-self.bots[bot_ind].state)
                    
                    if drone_distance <= 7 and self.bots[bot_ind].is_alive:
                        reward_list[agent_ind] += 100
                        self.bots[bot_ind].is_alive = False
                        self.bots[bot_ind].state[2] = 0.0


        if (not self.bots[0].is_alive) and (not self.bots[1].is_alive):
            done = True
            """
			for agent_ind in range(self.n_agents):
				reward_list[agent_ind] += 25
			"""

        if self.visualization:
            self.visualize()

        return self.get_observation(), reward_list/100, done, {}, [self.quadrotors[i].state for i in range(self.n_agents)], [self.bots[j].state for j in range(self.n_bots)]

    def get_observation(self):
        # conv_stack(batch,4,84,84) input_channel = 4
        # conv_stack(batch,16,84,84) = 5*agent1_pos + 5*agent2_pos + 5*uncertainty_grid + 1*obstacle_grid  
        # conv_stack(batch,5,4,84,84)

        # In the first 2 stacks, there will be position of agents (closest 4 grids to the agent will be 1, others will be 0)
        # In the third stack, there will be position of bots
        """
        conv_stack1 = np.zeros((self.N_frame*3, self.out_shape, self.out_shape))
        conv_stack2 = np.zeros((self.N_frame*3, self.out_shape, self.out_shape))
        obs_stack = np.zeros((self.n_agents, self.N_frame*3, self.out_shape, self.out_shape))

        for frame_ind in range(self.N_frame):
            # 0, 1, 2, 3, 4
            conv_stack1[frame_ind,:,:] = np.copy(self.agents_stacks[0][frame_ind])

        for frame_ind in range(self.N_frame):
            # 0, 1, 2, 3, 4
            conv_stack2[frame_ind,:,:] = np.copy(self.agents_stacks[1][frame_ind])

        for frame_ind in range(self.N_frame):
            # 5, 6, 7, 8, 9
            conv_stack1[self.N_frame+frame_ind,:,:] = np.copy(self.bots_stacks[0][frame_ind])
            conv_stack2[self.N_frame+frame_ind,:,:] = np.copy(self.bots_stacks[0][frame_ind])

        for frame_ind in range(self.N_frame):
            # 10, 11, 12, 13, 14
            conv_stack1[self.N_frame*2+frame_ind,:,:] = np.copy(self.bots_stacks[1][frame_ind])
            conv_stack2[self.N_frame*2+frame_ind,:,:] = np.copy(self.bots_stacks[1][frame_ind])

        obs_stack[0,:,:,:] = np.copy(conv_stack1)
        obs_stack[1,:,:,:] = np.copy(conv_stack2)
        """

        #state1 = [self.quadrotors[0].state[0], self.quadrotors[0].state[1], self.bots[0].state[0], self.bots[0].state[1], self.bots[1].state[0], self.bots[1].state[1]]
        #state2 = [self.quadrotors[1].state[0], self.quadrotors[1].state[1], self.bots[0].state[0], self.bots[0].state[1], self.bots[1].state[0], self.bots[1].state[1]]

        state = np.zeros((2,(self.n_bots+1)*3))

        for agent_ind in range(self.n_agents): 
            state[agent_ind][0:3] = self.quadrotors[agent_ind].state * self.quadrotors[agent_ind].is_alive
            for bot_ind in range(self.n_bots):
                # bot_ind:0 ->> 3,4,5 agent_state - bot1_state
                # bot_ind:1 ->> 6,7,8 agent_state - bot2 state
                state[agent_ind][(bot_ind*3)+3:(bot_ind*3)+6] = (self.quadrotors[agent_ind].state - self.bots[bot_ind].state)*self.bots[bot_ind].is_alive

        return np.array(state)


    def generate_agent_position(self):
        self.quadrotors = []

        for _ in range(0, self.n_agents):
            current_pos = [self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-12, high=-2)]
            state0 = [current_pos[0], current_pos[1], current_pos[2]]

            self.quadrotors.append(Drone(state0))

    def generate_bot_position(self):
        self.bots = []

        for _ in range(0, self.n_bots):
            current_pos = [self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-12, high=-2)]
            target_pos = [self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-12, high=-2)]

            state0 = [current_pos[0], current_pos[1], current_pos[2]]
            target_state0 = [target_pos[0], target_pos[1], target_pos[2]]

            self.bots.append(Bot(state0, target_state0))

    def check_collision(self):
        collision = False
        for agent_ind in range(self.n_agents):
            for other_agents_ind in range(self.n_agents):

                if agent_ind != other_agents_ind:
                    dist = np.linalg.norm(self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (dist <= 7):
                        collision = True

        for bot_ind in range(self.n_bots):
            for other_bots_ind in range(self.n_bots):

                if bot_ind != other_bots_ind:
                    dist = np.linalg.norm(self.bots[bot_ind].state-self.bots[other_bots_ind].state)

                    if (dist <= 7):
                        collision = True

        return collision

    def reset(self):
        self.generate_agent_position()
        self.generate_bot_position()
        self.iteration = 1

        collision = self.check_collision()

        if collision:
            return self.reset()
        else:
            pass

        return self.get_observation()


    def get_drone_stack(self, agent_ind):
        drone_closest_grids = self.get_closest_n_grids(self.quadrotors[agent_ind].state[0:2], self.neighbour_grids)

        drone_stack = np.zeros(self.uncertainty_grids.shape[0])
        drone_stack[drone_closest_grids] = 1
        drone_stack = np.reshape(drone_stack, (self.out_shape, self.out_shape))

        return drone_stack


    def get_bot_stack(self, bot_ind):
        if self.bots[bot_ind].is_alive:
            bot_closest_grids = self.get_closest_n_grids(self.bots[bot_ind].state[0:2], self.neighbour_grids)
            
            bot_stack = np.zeros(self.uncertainty_grids.shape[0])
            bot_stack[bot_closest_grids] = 1
            bot_stack = np.reshape(bot_stack, (self.out_shape, self.out_shape))
        else:
            bot_stack = np.zeros(self.uncertainty_grids.shape[0])
            bot_stack = np.reshape(bot_stack, (self.out_shape, self.out_shape))

        return bot_stack


    def get_drone_des_grid(self, drone_index, discrete_action):

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
        elif discrete_action == 4: #action=4, z += 1.0
            self.quadrotors[drone_index].state[2] += self.grid_res
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], -self.z_lim,  0)
        elif discrete_action == 5: #action=5, z -= 1.0
            self.quadrotors[drone_index].state[2] -= self.grid_res
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], -self.z_lim,  0)
        else:
            print ("Invalid discrete action!")

        drone_current_state = np.copy(self.quadrotors[drone_index].state)
        return drone_current_state


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


    def visualize(self, agent_pos_dict=None, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/black.png")
            fname2 = path.join(path.dirname(__file__), "assets/plane2.png")

            self.drone_transforms = []
            self.drones = []

            self.prey_transforms = []
            self.preys = []

            for i in range(self.n_agents):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(fname, 8., 8.))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(self.n_bots):
                self.prey_transforms.append(rendering.Transform())
                self.preys.append(rendering.Image(fname2, 8., 8.))
                self.preys[i].add_attr(self.prey_transforms[i])


        for i in range(self.n_bots):
            if self.bots[i].is_alive:
                self.viewer.add_onetime(self.preys[i])
                self.prey_transforms[i].set_translation(self.bots[i].state[0], self.bots[i].state[1])
                self.prey_transforms[i].set_rotation(self.bots[i].psid)
        
        for i in range(self.n_agents):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(self.quadrotors[i].state[0], self.quadrotors[i].state[1])
            self.drone_transforms[i].set_rotation(self.quadrotors[i].psi)
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
