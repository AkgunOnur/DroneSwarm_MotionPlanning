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

        print("curr_botPos",curr_botPos)

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

    def get_observation(self, agent_pos, bot_pos, n_agents, n_bots):

        state = np.zeros((self.n_agents,self.n_agents*3+self.n_bots*3))

        for agent_ind in range(self.n_agents):
            state[agent_ind][0:3] = self.quadrotors[agent_ind].state * self.quadrotors[agent_ind].is_alive
            for other_agents_ind in range(self.n_agents):
                if agent_ind != other_agents_ind:
                    state[agent_ind][(other_agents_ind*3):(other_agents_ind*3)+3] = self.quadrotors[other_agents_ind].state * self.quadrotors[other_agents_ind].is_alive

            for bot_ind in range(self.n_bots):
                state[agent_ind][(self.n_agents*3+bot_ind*3):(self.n_agents*3+bot_ind*3)+3] = (self.quadrotors[agent_ind].state - self.bots[bot_ind].state)*self.bots[bot_ind].is_alive

        return np.array(state)


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

    def reset(self, agent_pos, bot_pos):

        self.generate_agent_position(agent_pos)
        self.generate_bot_position(bot_pos)


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

    def get_bot_des_grid(self, bot_index):

        if self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] > 2:
            self.bots[bot_index].state[0] -= 0.3
        elif self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] < -2:
            self.bots[bot_index].state[0] += 0.3

        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] > 2:
            self.bots[bot_index].state[1] -= 0.3
        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] < -2:
            self.bots[bot_index].state[1] += 0.3

        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] > 2:
            self.bots[bot_index].state[2] -= 0.3
        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] < -2:
            self.bots[bot_index].state[2] += 0.3

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
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], -self.z_lim,  -1)
        elif discrete_action == 5: #action=5, z -= 1.0
            self.quadrotors[drone_index].state[2] -= self.grid_res
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], -self.z_lim,  -1)
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

    # client = airsim.MultirotorClient()
    # client.confirmConnection()

    with open('./agents_position/current_agents_pos.pkl', 'rb') as f:
        agent_pos = pickle.load(f)
    
    with open('./agents_position/current_bots_pos.pkl', 'rb') as f:
        bot_pos = pickle.load(f)

    n_agents = len(agent_pos)
    n_bots = len(bot_pos)

    agent_posL = []
    bot_posL = []

    # for agent_ind in range(n_agents):
    #     pos = client.simGetVehiclePose(vehicle_name=f"Drone{agent_ind+1}")
    #     agent_posL.append([pos.position.x_val, pos.position.y_val, pos.position.z_val])
    
    # print(agent_posL)

    # for bot_ind in range(n_bots):
    #     pos = client.simGetVehiclePose(vehicle_name=f"Drone{n_agents+bot_ind+1}")
    #     bot_posL.append([pos.position.x_val, pos.position.y_val, pos.position.z_val])

    HOST = '127.0.0.1'
    PORT = 9090

    virtual_env = VirEnv(n_agents=n_agents, n_bots=n_bots)
    virtual_env.reset(agent_pos, bot_pos)

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind((HOST, PORT))
    serverSocket.listen(5)
    clientsocket, clientAddress = serverSocket.accept()

    print("Connected by", clientAddress)
    print("Accepted a connection request from %s:%s"%(clientAddress[0], clientAddress[1]))

    while True:
        dataFromClient = pickle.loads(clientsocket.recv(1024))
        virtual_env.step(dataFromClient[0], dataFromClient[1])
