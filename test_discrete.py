import os
import torch
import torch.multiprocessing as mp


from sac_discrete.agent import SacdAgent, SharedSacdAgent
from sac_discrete.agent.sacd_decentralized import SacdAgent_Decentralized
from point_mass_formation_discrete import QuadrotorFormation



def main():
    n_agents = 2
    N_episodes = 10
    N_train = 5
    N_frame = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 25
    is_centralized = False
    visualization = True

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame, visualization=visualization, is_centralized=is_centralized)

    # Create the agent.
    if is_centralized:
        agent = SacdAgent(env=env, cuda=device, seed=seed)
        agent.target_critic.load_state_dict(torch.load("./models_centralized/final/target_critic.pth"))
        agent.target_critic.eval()
        agent.online_critic.load_state_dict(torch.load("./models_centralized/final/online_critic.pth"))
        agent.online_critic.eval()
        agent.policy.load_state_dict(torch.load("./models_centralized/final/policy.pth"))
        agent.policy.eval()
    else:
        agent = SacdAgent_Decentralized(env=env, cuda=device, seed=seed)
        for i in range(n_agents):
            agent.target_critic[i].load_state_dict(torch.load("./models_decentralized/final/target_critic_" + str(i+1) + ".pth"))
            agent.target_critic[i].eval()
            agent.online_critic[i].load_state_dict(torch.load("./models_decentralized/final/online_critic_" + str(i+1) + ".pth"))
            agent.online_critic[i].eval()
            agent.policy[i].load_state_dict(torch.load("./models_decentralized/final/policy_" + str(i+1) + ".pth"))
            agent.policy[i].eval()

    for i in range(N_episodes):
        agent.test_episode()



if __name__ == '__main__':
    main()
