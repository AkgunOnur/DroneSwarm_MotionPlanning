import gym
import numpy as np
import cv2
import time
import os
import torch

from PIL import Image
from gym import spaces
from gym.utils import seeding
from os import path
from gym.envs.classic_control import rendering
from collections import deque

# Angle Normalization function
def angle_normalize(angle):
    while(angle <= -np.pi):
        angle += 2 * np.pi
    while(angle > np.pi):
        angle -= 2 * np.pi
    return angle


class Drone:
    def __init__(self, width=0, height=0):
        self.v = 4
        self.x = 0
        self.y = 0
        self.psi = 0
        self.vx = 0
        self.vy = 0
        self.psi_dot = 0
        self.angle_between = 0
        self.reward = 0
        #self.zero_grid = np.zeros((width, height))
        self.is_alive = True


class Bot:
    def __init__(self, width=0, height=0):

        self.x_targetd = None
        self.y_targetd = None

        self.vd = 1
        self.xd = 0
        self.yd = 0
        self.psid = 0
        self.vxd = 0
        self.vyd = 0
        self.psi_dotd = 0
        self.angle_betweend = 0
        #self.zero_grid = np.zeros((width, height))
        self.is_alive = True


class Dubin(gym.Env):

    def __init__(self, args, target=[0.0, 0.0, 0.0], v=2.):
        super(Dubin, self).__init__()
        self.seed()
        self.num_bot = 2
        self.obs_state = "vector"
        self.stack = 5
        self.skip = 5
        self.target = target
        self.max_w = np.pi / 6
        self.dt = .05
        self.ix = 0
        self._cycle = 0
        self.capacity = self.stack
        self.width = 82
        self.height = 82
        self.window_w = 800
        self.window_h = 800
        self.reward = 0
        self.server = False
        self.episode_limit = 40000
        self.dim_actions = 1
        self.num_actions = 3
        self.n_action = 5

        self.bot1 = Bot(width=self.width, height=self.height)
        self.bot2 = Bot(width=self.width, height=self.height)
        self.bot_list = [self.bot1, self.bot2]
        self.n_bots = len(self.bot_list)

        self.agent1 = Drone(width=self.width, height=self.height)
        self.agent2 = Drone(width=self.width, height=self.height)
        self.agent_list = [self.agent1, self.agent2]
        self.n_agents = len(self.agent_list)
        print("N_AGENTS:", self.n_agents)
        print("N_BOTS:", self.n_bots)
        print("N_ACTIONS:", self.n_action)

        self.action_space = spaces.Box(low=-self.max_w,
                                        high=self.max_w, shape=(4,),
                                        dtype=np.float32)

        if self.obs_state == "vector":

            high = np.array([100., 100., 100., 100., 3., 3., 3., 3.], 
                            dtype=np.float32)
            low = np.array([0., 0., 0., 0., -3., -3., -3., -3.], 
                            dtype=np.float32)
            
            self.observation_space = spaces.Box(low=low,
                                                high=high,
                                                shape=(8,),
                                                dtype=np.float32)

        elif self.obs_state == "image":

            self.observation_space = spaces.Box(low=0, 
                                                high=255,  
                                                shape=(294,),
                                                dtype=np.uint8)
        
        self.viewer = None
        self.done = False

        self.agents_stacks = [deque([],maxlen=self.stack) for _ in range(self.n_agents)]
        self.bots_stacks = [deque([],maxlen=self.stack) for _ in range(self.n_bots)]

        for bot in self.bot_list:
            bot.x_targetd = self.np_random.uniform(low=10, high=72)
            bot.y_targetd = self.np_random.uniform(low=10, high=72)
        
        self.grid_res = 1.0
        self.x_lim = 82
        self.y_lim = 82
        self.neighbour_grids = 8

        X, Y = np.mgrid[0 : self.x_lim : self.grid_res, 
                        0 : self.y_lim : self.grid_res]

        # X: 82x82
        # Y: 82x82
        self.uncertainty_grids = np.vstack((X.flatten(), Y.flatten())).T
        # self.uncertainty_grids : 6724x2

    def get_img(self, screen):
        # 800x800x3
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        #screen = screen / 255.0
        screen = cv2.resize(screen, (self.width, self.height))
        # 82x82
        return np.array(screen)


    def get_stack(self):

        conv_stack = np.zeros((self.stack*(self.n_agents+self.n_bots), self.width, self.height))
        obs_stacks = np.zeros((self.n_agents, self.stack*(self.n_agents+self.n_bots), self.width, self.height))
        
        for agent_ind in range(self.n_agents):
            for frame_ind in range(self.stack):
                # agent_ind = 0, 0 1 2 3 4
                # agent_ind = 1, 5 6 7 8 9
                conv_stack[self.stack*agent_ind+frame_ind,:,:] = np.copy(self.agents_stacks[agent_ind][frame_ind])

        for bot_ind in range(self.n_bots):
            for frame_ind in range(self.stack):
                # bot_ind = 0, 10 11 12 13 14
                # bot_ind = 1, 15 16 17 18 19  
                conv_stack[self.stack*(self.n_agents+bot_ind)+frame_ind,:,:] = np.copy(self.bots_stacks[bot_ind][frame_ind])
        
        for i in range(self.n_agents):
            obs_stacks[i,:,:,:] = np.copy(conv_stack)
        #2x15x82x82
        return obs_stacks

    def get_closest_n_grids(self, current_pos, n):
        #print("current_pos", current_pos)
        #print("self.uncertainty_grids", self.uncertainty_grids.shape)
        differences = current_pos-self.uncertainty_grids
        #print("differences", differences)
        distances = np.sum(differences*differences,axis=1)
        #print("distances", distances.shape)
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        #print("sorted_indices",sorted_indices[0:n])
        
        return sorted_indices[0:n]

    def get_drone_stack(self, agent_ind):
        #print(self.agent_list[agent_ind].x, self.agent_list[agent_ind].y)
        drone_closest_grids = self.get_closest_n_grids([self.agent_list[agent_ind].x, self.agent_list[agent_ind].y], self.neighbour_grids)
        drone_stack = np.zeros(self.uncertainty_grids.shape[0])
        drone_stack[drone_closest_grids] = 1
        drone_stack = np.reshape(drone_stack, (self.width, self.height))

        return drone_stack

    def get_bot_stack(self, bot_ind):
        bot_closest_grids = self.get_closest_n_grids([self.bot_list[bot_ind].xd, self.bot_list[bot_ind].yd], self.neighbour_grids)
        bot_stack = np.zeros(self.uncertainty_grids.shape[0])
        bot_stack[bot_closest_grids] = 1
        bot_stack = np.reshape(bot_stack, (self.width, self.height))

        return bot_stack


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, test = None):
        self.reward1 = -2
        self.reward2 = -2

        for idx in range(self.skip):
            #time.sleep(0.03)
            ####################################################
            # ========== DUMMY PART ==========
            ####################################################
            # Generate new dummy target

            for bot in self.bot_list:
                if(np.abs(bot.x_targetd - bot.xd) <= 0.5 and np.abs(bot.y_targetd - bot.yd <= 0.5) and bot.is_alive):
                    bot.x_targetd = self.np_random.uniform(low= 10, high=70)
                    bot.y_targetd = self.np_random.uniform(low= 10, high=70)

                if bot.is_alive:
                    diff_Y = bot.y_targetd - bot.yd
                    diff_X = bot.x_targetd - bot.xd
                    psi_target = angle_normalize(np.arctan2(diff_Y, diff_X) - np.pi / 2)
                    bot.psi_dotd = angle_normalize(psi_target - bot.psid)
                    
                    bot.psid += (bot.psi_dotd * self.dt)
                    bot.vxd = bot.vd * np.cos(bot.psid + np.pi / 2)
                    bot.xd += bot.vxd * self.dt
                    bot.vyd = bot.vd * np.sin(bot.psid + np.pi / 2)
                    bot.yd += bot.vyd * self.dt
                    

            #####################################################
            # ========== AGENT PART ==========
            #####################################################

            #self.agent1.psi_dot = np.clip(u, -self.max_w, self.max_w)
            for i, agent in enumerate(self.agent_list):
                if int(u[0][i]) == 0:
                    agent.psi_dot = -self.max_w

                elif int(u[0][i]) == 1:
                    agent.psi_dot = self.max_w

                elif int(u[0][i]) == 2:
                    agent.psi_dot = 0

            # Update Plane Position
            
            for agent in self.agent_list:
                agent.psi = angle_normalize(agent.psi + agent.psi_dot * self.dt)
                agent.vx = agent.v * np.cos(agent.psi + np.pi / 2)
                agent.x += agent.vx * self.dt
                agent.vy = agent.v * np.sin(agent.psi + np.pi / 2)
                agent.y += agent.vy * self.dt
            

            # STATE
            ########################################################################### IMAGE #####################################################################
            if self.obs_state == "image":
                if test == None:
                    if idx == int(self.skip-1):
                        obs = self.render(mode='rgb_array')

                        for agent_ind in range(self.n_agents):
                            drone_stack = self.get_drone_stack(agent_ind)
                            self.agents_stacks[agent_ind].append(drone_stack)

                        for bot_ind in range(self.n_bots):
                            bot_stack = self.get_bot_stack(bot_ind)
                            self.bots_stacks[bot_ind].append(bot_stack)

                        #self.state = self.get_stack()
                else:
                    print("ajhdljasd")
                    obs = self.render(mode='rgb_array')
                    if idx == int(self.skip-1):
                        self.state = self.stack_image(obs)
            
            ########################################################################################################################################################
            
            ########################################################################### VECTOR #####################################################################
            elif self.obs_state == "vector":
                #print("VECTOR")
                if idx == int(self.skip-1):
                    self.render(mode='rgb_array')
                    self.state1 = np.array([(self.bot1.xd - self.agent1.x)*self.bot1.is_alive, (self.bot1.yd - self.agent1.y)*self.bot1.is_alive,
                                            (self.bot1.vxd - self.agent1.vx)*self.bot1.is_alive, (self.bot1.vyd - self.agent1.vy)*self.bot1.is_alive,
                                            (self.bot2.xd - self.agent1.x)*self.bot2.is_alive, (self.bot2.yd - self.agent1.y)*self.bot2.is_alive,
                                            (self.bot2.vxd - self.agent1.vx)*self.bot2.is_alive, (self.bot2.vyd - self.agent1.vy)*self.bot2.is_alive])

                    self.state2 = np.array([(self.bot1.xd - self.agent2.x)*self.bot1.is_alive, (self.bot1.yd - self.agent2.y)*self.bot1.is_alive,
                                            (self.bot1.vxd - self.agent2.vx)*self.bot1.is_alive, (self.bot1.vyd - self.agent2.vy)*self.bot1.is_alive,
                                            (self.bot2.xd - self.agent2.x)*self.bot2.is_alive, (self.bot2.yd - self.agent2.y)*self.bot2.is_alive,
                                            (self.bot2.vxd - self.agent2.vx)*self.bot2.is_alive, (self.bot2.vyd - self.agent2.vy)*self.bot2.is_alive])

            ###########################################################################################################################################################                            
        
        dist1 = np.sqrt((self.bot1.xd - self.agent1.x)**2 + (self.bot1.yd - self.agent1.y)**2)
        dist2 = np.sqrt((self.bot2.xd - self.agent1.x)**2 + (self.bot2.yd - self.agent1.y)**2)

        dist3 = np.sqrt((self.bot1.xd - self.agent2.x)**2 + (self.bot1.yd - self.agent2.y)**2)
        dist4 = np.sqrt((self.bot2.xd - self.agent2.x)**2 + (self.bot2.yd - self.agent2.y)**2)

        if (self.agent1.x >= 82 or self.agent1.y >= 82 or self.agent1.x <= 0 or self.agent1.y <= 0):
            self.done = True
            self.reward1 = -50

        if (self.agent2.x >= 82 or self.agent2.y >= 82 or self.agent2.x <= 0 or self.agent2.y <= 0):
            self.done = True
            self.reward2 = -50
        """
        if (dist1 <= 20 or dist3 <= 20) and self.bot1.is_alive:
            self.bot1.is_alive = False

        if (dist2 <= 20 or dist4 <= 20) and self.bot2.is_alive:
            self.bot2.is_alive = False

        if (not self.bot1.is_alive) and (not self.bot2.is_alive):
            self.reward1 = 50
            self.reward2 = 50
            self.done = True
        """

        if (dist1 <= 20 and self.bot1.is_alive):
            self.bot1.is_alive = False
            self.reward1 += 25

        if (dist3 <= 20 and self.bot1.is_alive):
            self.bot1.is_alive = False
            self.reward2 += 25

        if (dist2 <= 20 and self.bot2.is_alive):
            self.bot2.is_alive = False
            self.reward1 += 25

        if (dist4 <= 20 and self.bot2.is_alive):
            self.bot2.is_alive = False
            self.reward2 += 25

        if (not self.bot1.is_alive) and (not self.bot2.is_alive):
            self.done = True
            self.reward1 += 25
            self.reward2 += 25

        return np.array([self.state1, self.state2]) ,np.array([self.reward1/100,self.reward2/100]), self.done, {}
    
    def reset(self):
        self.agents_stacks = [deque([],maxlen=self.stack) for _ in range(self.n_agents)]
        self.bots_stacks = [deque([],maxlen=self.stack) for _ in range(self.n_bots)]
        
        # agent
        for agent in self.agent_list:
            agent.x = self.np_random.uniform(low=10, high=72)
            agent.y = self.np_random.uniform(low=10, high=72)
            agent.psi = self.np_random.uniform(low=-np.pi, high=np.pi)
        
        for bot in self.bot_list:
            # dummy
            bot.xd = self.np_random.uniform(low=10, high=72)
            bot.yd = self.np_random.uniform(low=10, high=72)
            bot.psid = self.np_random.uniform(low=-np.pi, high=np.pi)
            bot.is_alive = True
            #bot.zero_grid = np.zeros((self.width, self.height))
            
            # dummy targets
            bot.x_targetd = self.np_random.uniform(low=10, high=72)
            bot.y_targetd = self.np_random.uniform(low=10, high=72)


        # STATE
        if self.obs_state == "image":
            obs = self.render(mode='rgb_array')
            
            for i in range(self.stack):

                for agent_ind in range(self.n_agents):
                    drone_stack = self.get_drone_stack(agent_ind)
                    self.agents_stacks[agent_ind].append(drone_stack)

                for bot_ind in range(self.n_bots):
                    bot_stack = self.get_bot_stack(bot_ind)
                    self.bots_stacks[bot_ind].append(bot_stack)

            #self.state = self.get_stack()    

        elif self.obs_state == "vector":
            self.state1 = np.array([(self.bot1.xd - self.agent1.x)*self.bot1.is_alive, (self.bot1.yd - self.agent1.y)*self.bot1.is_alive,
                                            (self.bot1.vxd - self.agent1.vx)*self.bot1.is_alive, (self.bot1.vyd - self.agent1.vy)*self.bot1.is_alive,
                                            (self.bot2.xd - self.agent1.x)*self.bot2.is_alive, (self.bot2.yd - self.agent1.y)*self.bot2.is_alive,
                                            (self.bot2.vxd - self.agent1.vx)*self.bot2.is_alive, (self.bot2.vyd - self.agent1.vy)*self.bot2.is_alive])

            self.state2 = np.array([(self.bot1.xd - self.agent2.x)*self.bot1.is_alive, (self.bot1.yd - self.agent2.y)*self.bot1.is_alive,
                                            (self.bot1.vxd - self.agent2.vx)*self.bot1.is_alive, (self.bot1.vyd - self.agent2.vy)*self.bot1.is_alive,
                                            (self.bot2.xd - self.agent2.x)*self.bot2.is_alive, (self.bot2.yd - self.agent2.y)*self.bot2.is_alive,
                                            (self.bot2.vxd - self.agent2.vx)*self.bot2.is_alive, (self.bot2.vyd - self.agent2.vy)*self.bot2.is_alive])

        self.reward1 = 0
        self.reward2 = 0
        self.done = False
        
        #return self.get_stack()
        return np.array([self.state1, self.state2])

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_w, self.window_h)

            self.viewer.set_bounds(0,self.width,0,self.height)

            self.plane_transform1 = rendering.Transform()
            self.plane_transform2 = rendering.Transform()
            #self.target_transform = rendering.Transform()
            #self.target_transformd1 = rendering.Transform()
            #self.target_transformd2 = rendering.Transform()
            self.mp_transform1 = rendering.Transform()
            self.mp_transform2 = rendering.Transform()

            self.imgtrans = rendering.Transform()

            # Add plane pic
            
            if self.server:
                path = "/okyanus/users/deepdrone/multi-agent/centralized/Multi-Agent"
            else:
                path = os.getcwd()

            fname = "/assets/black.png"
            self.plane1 = rendering.Image(path + fname, 8., 8.)
            self.plane1.add_attr(self.plane_transform1)

            #fname2 = "/assets/dr.png"
            self.plane2 = rendering.Image(path + fname, 8., 8.)
            self.plane2.add_attr(self.plane_transform2)

            # Added runway pic
            """
            fname2 = "/assets/runway.png"
            self.runway = rendering.Image(fname2, 5., 5.)
            self.runway.add_attr(self.target_transform)
            """
            # Added runway pic
            
            fname3 = "/assets/gplane.png"
            self.mp1 = rendering.Image(path + fname3, 8., 8.)
            self.mp1.add_attr(self.mp_transform1)

            self.mp2 = rendering.Image(path + fname3, 8., 8.)
            self.mp2.add_attr(self.mp_transform2)

        self.viewer.add_onetime(self.plane1)
        self.viewer.add_onetime(self.plane2)

        if self.bot1.is_alive:
            self.viewer.add_onetime(self.mp1)
        if self.bot2.is_alive:
            self.viewer.add_onetime(self.mp2)

        # Add plane position and orientation
        self.plane_transform1.set_translation(self.agent1.x, self.agent1.y)
        self.plane_transform1.set_rotation(self.agent1.psi)

        self.plane_transform2.set_translation(self.agent2.x, self.agent2.y)
        self.plane_transform2.set_rotation(self.agent2.psi)

        # Add dummy points
        """
        self.target_transformd1.set_translation(
            self.bot1.x_targetd, self.bot1.y_targetd)

        self.target_transformd2.set_translation(
                    self.bot2.x_targetd, self.bot2.y_targetd)
        """
        # Add runway position and orientation
        if self.bot1.is_alive:
            self.mp_transform1.set_translation(self.bot1.xd, self.bot1.yd)
            self.mp_transform1.set_rotation(self.bot1.psid)
        if self.bot2.is_alive:
            self.mp_transform2.set_translation(self.bot2.xd, self.bot2.yd)
            self.mp_transform2.set_rotation(self.bot2.psid)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass