import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pickle

def plot_trajectory(agent_p, agent_list):
    n_agents = 5# len(agent_list)
    numDataPoints = len(agent_p)

    fig = plt.figure()
    ax = Axes3D(fig)
    
    color_list = ['g', 'b', 'r', 'y', 'm']
    
    line = [None for _ in range(n_agents)]
    # agentDots = [None for _ in range(n_agents)]
    xdata = [[] for _ in range(n_agents)]
    ydata = [[] for _ in range(n_agents)]
    zdata = [[] for _ in range(n_agents)]


    # for n_a in range(n_agents):
    #     agentDots[n_a] = ax.scatter3D(agent_p[0][n_a][0], agent_p[0][n_a][1], agent_p[0][n_a][2], lw=4, c=color_list[n_a])

    for point_idx in range(len(agent_p)):
        for n_a in range(n_agents):
            xdata[n_a].append(agent_p[point_idx][n_a][0])
            ydata[n_a].append(agent_p[point_idx][n_a][1])
            zdata[n_a].append(agent_p[point_idx][n_a][2])

    for n_a in range(n_agents):
        line[n_a] = plt.plot(xdata[n_a], ydata[n_a], zdata[n_a], lw=2, c=color_list[n_a])[0]
        
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((0, 6))

    ax.set_xlabel('X - Axis')
    ax.set_ylabel('Y - Axis')
    ax.set_zlabel('Z - Axis')
    ax.set_title('Trajectory of agents')

    line_ani = animation.FuncAnimation(fig, animate, frames=numDataPoints, fargs=(xdata, ydata, zdata, line, n_agents), interval=50, blit=False)

    plt.show()

def animate(num, x, y, z, line, n_agents):
    for n_a in range(n_agents):
        line[n_a].set_data(np.array([x[n_a][:num], y[n_a][:num]]))    
        line[n_a].set_3d_properties(z[n_a][:num])    

    return line
    

if __name__ == "__main__":
    with open('./agents_position/agents_positions_planner.pkl', 'rb') as f:
        total_pos_list = pickle.load(f)
    total_pos_list = np.asarray(total_pos_list).squeeze(axis=0)
    episode_pos_list = total_pos_list[:,:,0:3]
    print("shape: ", episode_pos_list.shape)

    plot_trajectory(episode_pos_list, episode_pos_list.shape[1])