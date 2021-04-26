import numpy as np
import pickle
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# This import registers the 3D projection, but is otherwise unused.
# from random import randint
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib.colors


x_lim = 20  # grid x limit
y_lim = 20  # grid y limit
z_lim = 6  # grid z limit
res = 1.0  # resolution for grids
out_shape = 164  # width and height for uncertainty matrix
dist = 5.0  # distance threshold

X, Y, Z = np.mgrid[-x_lim:x_lim + 0.1:res,
                   -y_lim:y_lim + 0.1:res,
                   0:z_lim + 0.1: 2 * res]
uncertainty_grids = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
grid_visits = np.zeros((uncertainty_grids.shape[0], ))
uncertainty_values = np.random.uniform(
    low=0.95, high=1.0, size=(uncertainty_grids.shape[0],))

#######################################

# prepare some coordinates
voxels = np.zeros((41, 41, 7))  # np.mgrid[-20:20:41j, -20:20:41j, 0:15:16j]
voxels[:, :, :] = False
# set the colors of each object
x, y, z = np.indices(np.array(voxels.shape) + 1)

##########################################
total_grids = voxels.shape[0] * voxels.shape[1] * 4  # voxels.shape[2]


def get_closest_n_grids(current_pos, n):
    differences = current_pos - uncertainty_grids
    distances = np.sum(differences * differences, axis=1)
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
    return sorted_indices[0:n]


episode = 0
n_agents = 2


############################
def update(R):
    # for episode in range(1):
    # print(len(agent_pos_over_episodes))
    for episode_pos in total_pos_list:
        print("episode_pos: ", episode_pos.shape)
        for i in range(R):
            # print(i)
            for agent_ind in range(episode_pos.shape[2]):
                agent_pos = episode_pos[:, i, agent_ind]
                indices = get_closest_n_grids(agent_pos, 8)
                # print ("Episode {4}, Agent {0} X:{1:.4}, Y:{2:.4}, Z:{3:.4}".format(agent_ind+1, agent_pos[0], agent_pos[1], agent_pos[2], episode+1))
                for a in range(uncertainty_grids[indices].shape[0]):
                    voxels[:, :, int(uncertainty_grids[indices][a, 2] - 1)][int(
                        uncertainty_grids[indices][a, 0]) + 20, int(uncertainty_grids[indices][a, 1]) + 20] = True

    scanned_grids = np.sum(voxels)
    # Count number of scanned cubes
    print("Map coverage: ", 100 * scanned_grids / total_grids)


def update_combined(R):
    # for episode in range(1):
    # print(len(agent_pos_over_episodes))
    for i in range(R):  # episode_pos_list.shape
        print(i)
        for agent_ind in range(2):
            for e in range(5):
                #print("agent_ind: ", agent_ind)
                #agent_pos = episode_pos_list[:, i, agent_ind]
                if isinstance(episode_pos_list[i], list):
                    pozisyon = episode_pos_list[i][agent_ind][e]
                    #print("POZISYON1: ", pozisyon)
                else:
                    pozisyon = episode_pos_list[i][:, e, agent_ind]
                    #print("POZISYON2: ", pozisyon)
                agent_pos = pozisyon
                indices = get_closest_n_grids(agent_pos, 8)
                # print ("Episode {4}, Agent {0} X:{1:.4}, Y:{2:.4}, Z:{3:.4}".format(agent_ind+1, agent_pos[0], agent_pos[1], agent_pos[2], episode+1))
                for a in range(uncertainty_grids[indices].shape[0]):
                    voxels[:, :, int(uncertainty_grids[indices][a, 2] - 1)][int(
                        uncertainty_grids[indices][a, 0]) + 20, int(uncertainty_grids[indices][a, 1]) + 20] = True

    scanned_grids = np.sum(voxels)
    # Count number of scanned cubes
    print("Map coverage: ", 100 * scanned_grids / total_grids)


def get_map_coverage():
    average_val_list = []

    with open('agents_positions_4.pkl', 'rb') as f:
        total_pos_list = pickle.load(f)

    for episode in range(len(total_pos_list)):
        grid_visits = np.zeros((uncertainty_grids.shape[0], ))
        episode_pos_list = total_pos_list[episode]
        for iteration in range(episode_pos_list.shape[1]):
            for agent_ind in range(episode_pos_list.shape[2]):
                agent_pos = episode_pos_list[:, iteration, agent_ind]
                indices = get_closest_n_grids(agent_pos, 8)
                grid_visits[indices] = 1

        visited_grids = len(np.where(grid_visits == 1)[0])
        average_val = 100 * visited_grids / len(grid_visits)
        average_val_list.append(average_val)
        print("Episode: {0} Map coverage: {1:.4}".format(
            episode + 1, average_val))

    print("Average map coverage: ", np.mean(average_val_list))
    print("Std map coverage: ", np.std(average_val_list))

    # for episode in range(len(total_pos_list)):
    #     episode_pos_list = total_pos_list[episode]
    #     for episode_pos in episode_pos_list:
    #         print ("episode_pos: ", episode_pos.shape)
    #         N_iteration = episode_pos.shape[1]
    #         for i in range(N_iteration):
    #             # print(i)
    #             for agent_ind in range(episode_pos.shape[2]):
    #                 agent_pos = episode_pos[:, i, agent_ind]
    #                 indices = get_closest_n_grids(agent_pos, 8)
    #                 grid_visits[indices] = 1
    #                 # print ("Episode {4}, Agent {0} X:{1:.4}, Y:{2:.4}, Z:{3:.4}".format(agent_ind+1, agent_pos[0], agent_pos[1], agent_pos[2], episode+1))
    #                 # for a in range(uncertainty_grids[indices].shape[0]):
    #                 #     voxels[:, :, int(uncertainty_grids[indices][a, 2] - 1)][int(
    #                 #         uncertainty_grids[indices][a, 0]) + 20, int(uncertainty_grids[indices][a, 1]) + 20] = True


if __name__ == '__main__':
    # R = 7500
    # update(R)
    get_map_coverage()
