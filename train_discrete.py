import os
import torch
import torch.multiprocessing as mp


from sac_discrete.agent import SacdAgent, SharedSacdAgent
from sac_discrete.agent.sacd_decentralized import SacdAgent_Decentralized
from point_mass_formation_discrete import QuadrotorFormation


def main():
    n_agents = 2
    N_frame = 5
    Max_iteration = 6000
    Eval_interval = 500
    N_episodes = 50000
    Batch_size = 256
    Memory_Size = 150000
    Training_Start = 200

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    seed = 27
    is_centralized = True
    visualization = False

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame,
                             visualization=visualization, is_centralized=is_centralized)

    # Create the agent.
    if is_centralized:
        agent = SacdAgent(env=env, batch_size=Batch_size, memory_size=Memory_Size, start_steps=Training_Start, update_interval=4, target_update_interval=12,
                          use_per=True, dueling_net=True, max_episode_steps=N_episodes, eval_interval=Eval_interval, max_iteration_steps=Max_iteration,
                          device=device, seed=seed)
    else:
        agent = SacdAgent_Decentralized(env=env, batch_size=Batch_size, memory_size=Memory_Size, start_steps=Training_Start, update_interval=4, target_update_interval=12,
                                        use_per=True, dueling_net=True, max_episode_steps=N_episodes, eval_interval=Eval_interval, max_iteration_steps=Max_iteration,
                                        device=device, seed=seed)

    agent.train_episode()


if __name__ == '__main__':
    main()
