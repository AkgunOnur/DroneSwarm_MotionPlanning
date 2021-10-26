import gym
from gym import spaces, error, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from os import path
import itertools
import airsim
import pickle
import socket

from quadrotor_dynamics import Quadrotor, Drone, Bot
from numpy.random import uniform
from collections import deque

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class VirEnv(gym.Env):

    def __init__(self, n_agents=2, n_bots = 2, N_frame=5, visualization=True, is_centralized = False, moving_target = False):

        self.seed()
        self.n_action = 6
        self.observation_dim = 4
        self.dim_actions = 1
        self.n_agents = n_agents
        self.n_bots = n_bots
        self.visualization = visualization
        self.is_centralized = is_centralized
        self.moving_target = moving_target
        self.action_dict = {0:"Xp", 1:"Xn", 2:"Yp", 3:"Yn"}

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

    def step(self, curr_agentPos, curr_botPos):

        # with open('./agents_position/current_agents_pos.pkl', 'rb') as f:
        #     curr_agentPos = pickle.load(f)
    
        # with open('./agents_position/current_bots_pos.pkl', 'rb') as f:
        #     curr_botPos = pickle.load(f)
        
        # Environment Step

        for agent_ind in range(self.n_agents):
            self.quadrotors[agent_ind].state[0] = curr_agentPos[agent_ind][0]
            self.quadrotors[agent_ind].state[1] = curr_agentPos[agent_ind][1]
            self.quadrotors[agent_ind].state[2] = curr_agentPos[agent_ind][2]

            if self.quadrotors[agent_ind].state[2] == 0.0 and self.quadrotors[agent_ind].is_alive:
                self.quadrotors[agent_ind].is_alive = False

        ########################### CLIENT #############################
        # for agent_ind in range(self.n_agents):
        #     self.quadrotors[agent_ind].state[0] = client.simGetVehiclePose(vehicle_name=f"Drone{agent_ind+1}").position.x_val
        #     self.quadrotors[agent_ind].state[1] = client.simGetVehiclePose(vehicle_name=f"Drone{agent_ind+1}").position.y_val
        #     self.quadrotors[agent_ind].state[2] = client.simGetVehiclePose(vehicle_name=f"Drone{agent_ind+1}").position.z_val

        #     print(agent_ind, self.quadrotors[agent_ind].state)
        
        # for bot_ind in range(self.n_bots):
        #     self.bots[bot_ind].state[0] = client.simGetVehiclePose(vehicle_name=f"Drone{self.n_agents+bot_ind+1}").position.x_val
        #     self.bots[bot_ind].state[1] = client.simGetVehiclePose(vehicle_name=f"Drone{self.n_agents+bot_ind+1}").position.y_val
        #     self.bots[bot_ind].state[2] = client.simGetVehiclePose(vehicle_name=f"Drone{self.n_agents+bot_ind+1}").position.z_val

        #################################################################

        for bot_ind in range(self.n_bots):
            if self.bots[bot_ind].is_alive:
                self.bots[bot_ind].state[0] = curr_botPos[bot_ind][0]
                self.bots[bot_ind].state[1] = curr_botPos[bot_ind][1]
                self.bots[bot_ind].state[2] = curr_botPos[bot_ind][2]
                
            if self.bots[bot_ind].state[2] == 0.0 and self.bots[bot_ind].is_alive:
                self.bots[bot_ind].is_alive = False

        if self.visualization:
            self.visualize()



    def generate_agent_position(self, agent_pos):
        self.quadrotors = []

        for cur_pos in agent_pos:
            state0 = [cur_pos[0], cur_pos[1], cur_pos[2]]

            self.quadrotors.append(Drone(state0))

    def generate_bot_position(self, bot_pos):
        self.bots = []

        for cur_pos in bot_pos:
            state0 = [cur_pos[0], cur_pos[1], cur_pos[2]]
            target_state0 = []

            self.bots.append(Bot(state0, target_state0))

    def reset(self, agent_pos, bot_pos):
        self.generate_agent_position(agent_pos)
        self.generate_bot_position(bot_pos)


    def visualize(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/dr.png")
            fname2 = path.join(path.dirname(__file__), "assets/plane2.png")

            self.drone_transforms = []
            self.drones = []

            self.prey_transforms = []
            self.preys = []

            for i in range(self.n_agents):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(fname, 5., 5.))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(self.n_bots):
                self.prey_transforms.append(rendering.Transform())
                self.preys.append(rendering.Image(fname2, 5., 5.))
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
        
        return self.viewer.render(return_rgb_array = mode =='rgb_array')

if __name__ == "__main__":

    with open('./agents_position/agents_positions.pkl', 'rb') as f:
        agent_pos = pickle.load(f)
    
    with open('./agents_position/bots_positions.pkl', 'rb') as f:
        bot_pos = pickle.load(f)

    n_agents = len(agent_pos[0][0])
    n_bots = len(bot_pos[0][0])

    HOST = '127.0.0.1'
    PORT = 9090

    virtual_env = VirEnv(n_agents=n_agents, n_bots=n_bots)

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind((HOST, PORT))
    serverSocket.listen(5)
    clientsocket, clientAddress = serverSocket.accept()

    print("Connected by", clientAddress)
    print("Accepted a connection request from %s:%s"%(clientAddress[0], clientAddress[1]))

    idx = 0
    while True:

        dataFromClient = pickle.loads(clientsocket.recv(1024))

        if idx == 0:
            virtual_env.reset(dataFromClient[0], dataFromClient[1])

        virtual_env.step(dataFromClient[0], dataFromClient[1])

        idx += 1
