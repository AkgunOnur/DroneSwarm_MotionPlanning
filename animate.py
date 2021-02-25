import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
# This import registers the 3D projection, but is otherwise unused.
from random import randint
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.colors

import pickle

x_lim = 20  # grid x limit
y_lim = 20  # grid y limit
z_lim = 15  # grid z limit
res = 1.0  # resolution for grids
out_shape = 164  # width and height for uncertainty matrix
dist = 5.0  # distance threshold

X, Y, Z = np.mgrid[-x_lim:x_lim + 0.1:res, -
                   y_lim:y_lim + 0.1:res, 0:z_lim + 0.1:res*2]
uncertainty_grids = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
uncertainty_values = np.random.uniform(
    low=0.95, high=1.0, size=(uncertainty_grids.shape[0],))
grid_visits = np.zeros((uncertainty_grids.shape[0], ))


#######################################

# prepare some coordinates
voxels = np.zeros((41, 41, 7))  # np.mgrid[-20:20:41j, -20:20:41j, 0:15:16j]
voxels[:, :, :] = False
# set the colors of each object
x, y, z = np.indices(np.array(voxels.shape) + 1)

##########################################
# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(-20.0, 20.0)
ax.set_ylim3d(-20.0, 20.0)
ax.set_zlim3d(0.0, 6.0)



total_grids = voxels.shape[0] * voxels.shape[1] * voxels.shape[2] // 2



N_episode = 1 #len(total_pos_list)


def get_closest_n_grids(current_pos, n):
    differences = current_pos-uncertainty_grids
    distances = np.sum(differences*differences,axis=1)
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
    
    return sorted_indices[0:n]


def voxel_drawing():
    with open('agents_positions.pkl', 'rb') as f:
        total_pos_list = pickle.load(f)
        
    episode = 0

    episode_pos_list = total_pos_list[episode]
    for iteration in range(episode_pos_list.shape[1]):
        for agent_ind in range(episode_pos_list.shape[2]):
            agent_pos = episode_pos_list[:, iteration, agent_ind]
            indices = get_closest_n_grids(agent_pos, 8)
            # print ("Episode {4}, Agent {0} X:{1:.4}, Y:{2:.4}, Z:{3:.4}".format(agent_ind+1, agent_pos[0], agent_pos[1], agent_pos[2], episode+1))
            for a in range(uncertainty_grids[indices].shape[0]):
                voxels[:, :, int(uncertainty_grids[indices][a, 2])][int(
                    uncertainty_grids[indices][a, 0]) + 20, int(uncertainty_grids[indices][a, 1]) + 20] = True

        ax.voxels(x - 20, y - 20, z, voxels, edgecolor='k')
        plt.draw()
        plt.pause(0.02)
    
    scanned_grids = np.sum(voxels)
    # Count number of scanned cubes
    print("Map coverage: ", 100 * scanned_grids / total_grids)


def circle_drawing():
    with open('agents_positions.pkl', 'rb') as f:
        total_pos_list = pickle.load(f)
        
    episode = 0

    episode_pos_list = total_pos_list[episode]
    for iteration in range(episode_pos_list.shape[1]):
        for agent_ind in range(episode_pos_list.shape[2]):
            color = ['red', 'green']
            agent_pos = episode_pos_list[:, iteration, agent_ind]
            indices = get_closest_n_grids(agent_pos, 8)
            # print ("Episode {4}, Agent {0} X:{1:.4}, Y:{2:.4}, Z:{3:.4}".format(agent_ind+1, agent_pos[0], agent_pos[1], agent_pos[2], episode+1))
            for a in range(uncertainty_grids[indices].shape[0]):
                ax.scatter(uncertainty_grids[indices][a, 0], uncertainty_grids[indices][a, 1], uncertainty_grids[indices][a, 2], 
                        color=color[agent_ind], s=50)
                plt.draw()
                plt.pause(0.02)

    
    # scanned_grids = np.sum(voxels)
    # # Count number of scanned cubes
    # print("Map coverage: ", 100 * scanned_grids / total_grids)





# anim = FuncAnimation(fig, update, frames=np.arange(
#     0, 5, 1), repeat=False, fargs=(fig, ax))
# anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)

if __name__ == "__main__":
    circle_drawing()

