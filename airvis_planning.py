import gym
from gym import spaces, error, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from os import path
import itertools
import random
import pdb
from quadrotor_dynamics import Quadrotor
from numpy.random import uniform
from time import sleep
from collections import deque
import warnings
import pickle
import socket

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

class VirQuadEnv(gym.Env):
    def __init__(self, n_agents=5, N_frame=5, visualization=True):
        warnings.filterwarnings('ignore')
        # number of actions per agent which are desired positions and yaw angle
        self.seed()
        self.n_agents = n_agents
        self.visualization = visualization

        #state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.quadrotors = []
        self.viewer = None
        self.x_lim = 20.0
        self.y_lim = 20.0
        self.z_lim = 6.0

        self.action_list = []
        for p in itertools.product([0,1,2,3,4,5], repeat=2):
            self.action_list.append(p)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, curr_agentPos):
        for agent_ind in range(self.n_agents):
            self.quadrotors[agent_ind].state[0] = curr_agentPos[agent_ind][0]
            self.quadrotors[agent_ind].state[1] = curr_agentPos[agent_ind][1]
            self.quadrotors[agent_ind].state[2] = curr_agentPos[agent_ind][2]


        if self.visualization:
            self.visualize()

    def reset(self, agent_pos):
        print("agent_pos: ", agent_pos)
        self.quadrotors = []
        for agent_ind in range(0, self.n_agents):
            state0 = [agent_pos[agent_ind][0], agent_pos[agent_ind][1], agent_pos[agent_ind][2],
                      0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.quadrotors.append(Quadrotor(state0))

    def visualize(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/dr.png")

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
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == "__main__":
    with open('./agents_position/agents_positions_planner.pkl', 'rb') as f:
        agent_pos = pickle.load(f)

    n_agents = len(agent_pos[0][0])
    print("nagents:", n_agents)

    HOST = '127.0.0.1'
    PORT = 9090

    virtual_env = VirQuadEnv(n_agents=n_agents)

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
            virtual_env.reset(dataFromClient[0])
        virtual_env.step(dataFromClient[0])

        idx += 1