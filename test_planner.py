import os
import torch
import torch.multiprocessing as mp
import pickle
from planner import *


from sac_discrete.agent import SacdAgent, SharedSacdAgent
from sac_discrete.agent.sacd_decentralized import SacdAgent_Decentralized
from point_mass_formation_discrete import QuadrotorFormation


def main():
    n_agents = 2
    N_frame = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 25
    is_centralized = True
    visualization = True
    start_steps = 100
    N_iteration = 5
    update_interval = 5
    eval_interval = 100
    best_reward = -np.Inf
    model_dir = '/okyanus/users/deepdrone/DroneSwarm_MotionPlanning/models_planner'

    file_name = "best"
    episode_number = 20000
    iteration_number = 400
    total_pos_list = []

    # Create environments.
    env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame, visualization=visualization, is_centralized=is_centralized)
    # Create the agents
    agent_c = SacdAgent(env=env, batch_size=128, memory_size=150000, start_steps=200, update_interval=4, target_update_interval=12,
                          use_per=True, dueling_net=True, max_episode_steps=100000, eval_interval=500, max_iteration_steps=250,
                          device=device, seed=seed)
    agent_d = SacdAgent_Decentralized(env=env, batch_size=128, memory_size=150000, start_steps=200, update_interval=4, target_update_interval=12,
                          use_per=True, dueling_net=True, max_episode_steps=100000, eval_interval=500, max_iteration_steps=6000,
                          device=device, seed=seed)

    if file_name == "best":
        agent_c.policy.load_state_dict(torch.load("./models_centralized/" +  file_name + "/policy_1.pth"))
        agent_c.policy.eval()

        for i in range(n_agents):
            agent_d.policy[i].load_state_dict(torch.load("./models_decentralized/" + file_name + "/policy_" + str(i+1) + "_1.pth"))
            agent_d.policy[i].eval()
    else:
        agent_c.policy.load_state_dict(torch.load("./models_centralized/" +  file_name + "/policy_" + str(episode_number) + ".pth"))
        agent_c.policy.eval()

        for i in range(n_agents):
            agent_d.policy[i].load_state_dict(torch.load("./models_decentralized/" + file_name + "/policy_" + str(i+1) + "_" + str(episode_number) + ".pth"))
            agent_d.policy[i].eval()



    dqn = DQN()
    dqn.eval_net.load_state_dict(torch.load("./models_planner/" + file_name + "/policy_1.pth"))
    dqn.eval_net.eval()
    planner_dict = ["Cent", "Decent"]
    

    for i_episode in range(episode_number):
        agent_obs = env.reset()
        episode_reward = 0
        planner_output = []
        for i_iteration in range(iteration_number):
            action = dqn.choose_action(agent_obs)
            planner_output.append(planner_dict[action])
            if action == 0: #centralized option
                next_agent_obs, reward, done = agent_c.test_planner(agent_obs, N_iteration)
            else: #decentralized option
                next_agent_obs, reward, done = agent_d.test_planner(agent_obs, N_iteration)

            episode_reward += reward

            # if done:
            #     break

            agent_obs = next_agent_obs

            print ("Executed method: ", planner_dict[action])

        print('\n Episode: ', i_episode, '| Episode_reward: ', round(episode_reward, 2))
        planner_output = np.array(planner_output)
        print ("Centralized distribution: ", np.sum(planner_output == planner_dict[0]) / len(planner_output))
        print ("Decentralized distribution: ", np.sum(planner_output == planner_dict[1]) / len(planner_output))


if __name__ == '__main__':
    main()
