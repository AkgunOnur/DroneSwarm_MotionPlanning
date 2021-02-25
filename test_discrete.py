import os
import torch
import torch.multiprocessing as mp
import pickle


from sac_discrete.agent import SacdAgent, SharedSacdAgent
from sac_discrete.agent.sacd_decentralized import SacdAgent_Decentralized
from point_mass_formation_discrete import QuadrotorFormation



def main():
    n_agents = 2
    N_test_episodes = 1
    N_frame = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 25
    is_centralized = True
    visualization = False

    file_name = "best"
    episode_number = 18000
    total_pos_list = []

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame, visualization=visualization, is_centralized=is_centralized)

    # Create the agent.
    if is_centralized:
        agent = SacdAgent(env=env, batch_size=128, memory_size=150000, start_steps=200, update_interval=4, target_update_interval=12,
                          use_per=True, dueling_net=True, max_episode_steps=100000, eval_interval=500, max_iteration_steps=250,
                          device=device, seed=seed)
        if file_name == "best":
            agent.target_critic.load_state_dict(torch.load("./models_centralized/" + file_name + "/target_critic_1.pth"))
            agent.target_critic.eval()
            agent.online_critic.load_state_dict(torch.load("./models_centralized/" + file_name + "/online_critic_1.pth"))
            agent.online_critic.eval()
            agent.policy.load_state_dict(torch.load("./models_centralized/" +  file_name + "/policy_1.pth"))
        else:
            agent.target_critic.load_state_dict(torch.load("./models_centralized/" + file_name + "/target_critic_" + str(episode_number) + ".pth"))
            agent.target_critic.eval()
            agent.online_critic.load_state_dict(torch.load("./models_centralized/" + file_name + "/online_critic_" + str(episode_number) + ".pth"))
            agent.online_critic.eval()
            agent.policy.load_state_dict(torch.load("./models_centralized/" +  file_name + "/policy_" + str(episode_number) + ".pth"))

        agent.policy.eval()
    else:
        agent = SacdAgent_Decentralized(env=env, batch_size=128, memory_size=150000, start_steps=200, update_interval=4, target_update_interval=12,
                          use_per=True, dueling_net=True, max_episode_steps=100000, eval_interval=500, max_iteration_steps=6000,
                          device=device, seed=seed)
        for i in range(n_agents):
            agent.target_critic[i].load_state_dict(torch.load("./models_decentralized/" + file_name + "/target_critic_" + str(i+1) + ".pth"))
            agent.target_critic[i].eval()
            agent.online_critic[i].load_state_dict(torch.load("./models_decentralized/" + file_name + "/online_critic_" + str(i+1) + ".pth"))
            agent.online_critic[i].eval()
            agent.policy[i].load_state_dict(torch.load("./models_decentralized/" + file_name + "/policy_" + str(i+1) + ".pth"))
            agent.policy[i].eval()

    for i in range(N_test_episodes):
        episode_pos_list = agent.test_episode()
        total_pos_list.append(episode_pos_list)

        with open('agents_positions.pkl', 'wb') as f:
            pickle.dump(total_pos_list, f)



if __name__ == '__main__':
    main()
