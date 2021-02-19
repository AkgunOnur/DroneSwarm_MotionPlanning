import os
import torch

from sac_discrete.agent import SacdAgent, SharedSacdAgent
from point_mass_formation_discrete import QuadrotorFormation



def main():
    n_agents = 2
    N_episodes = 50000
    N_iteration = 250
    batch_size = 256
    N_train = 5
    N_frame = 5
    train_episode_modulo = 10
    exploration_episode = 50
    device = "cpu"
    seed = 25

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame, visualization=True)

    # Create the agent.
    Agent = SacdAgent
    agent = Agent(env=env, cuda=device, seed=seed)
    agent.train_episode()


if __name__ == '__main__':
    main()
