import os
import torch
import torch.multiprocessing as mp


from sac_discrete.agent import SacdAgent, SharedSacdAgent
from sac_discrete.agent.sacd_decentralized import SacdAgent_Decentralized
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 25
    is_centralized = False
    visualization = False

    num_processes = 2

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame, visualization=visualization, is_centralized=is_centralized)

    # Create the agent.
    if is_centralized:
        agent = SacdAgent(env=env, cuda=device, seed=seed)
    else:
        agent = SacdAgent_Decentralized(env=env, cuda=device, seed=seed)
    
    agent.train_episode()


if __name__ == '__main__':
    main()
