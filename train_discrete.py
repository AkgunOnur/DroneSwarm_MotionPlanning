import os
import torch
import torch.multiprocessing as mp


from sac_discrete.agent import SacdAgent, SharedSacdAgent
from sac_discrete.agent.sacd_decentralized import SacdAgent_Decentralized
from point_mass_formation_discrete import QuadrotorFormation


def main():
    n_agents = 2
    N_train = 5
    N_frame = 5
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    seed = 25
    is_centralized = True
    visualization = False

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame,
                             visualization=visualization, is_centralized=is_centralized)

    # Create the agent.
    if is_centralized:
        agent = SacdAgent(env=env, n_agents=n_agents, N_frame=N_frame, batch_size=128,
                          memory_size=150000, start_steps=200, update_interval=4, target_update_interval=12,
                          use_per=True, dueling_net=True, max_episode_steps=100000, eval_interval=500, max_iteration_steps=250,
                          device=device, seed=seed)
    else:
        agent = SacdAgent_Decentralized(env=env, n_agents=n_agents, N_frame=N_frame, batch_size=128,
                                        memory_size=150000, start_steps=200, update_interval=4, target_update_interval=12,
                                        use_per=True, dueling_net=True, max_episode_steps=100000, eval_interval=500, max_iteration_steps=250,
                                        device=device, seed=seed)

    agent.train_episode()


if __name__ == '__main__':
    main()
