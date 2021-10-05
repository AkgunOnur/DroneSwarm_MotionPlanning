import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pickle

def animate(num, x, y, z, line, n_agents, bot_p, redDots):

    for n_a in range(n_agents):
        line[n_a].set_data(np.array([x[n_a][:num], y[n_a][:num]]))    
        line[n_a].set_3d_properties(z[n_a][:num])

    print(bot_p)
    for n_b in range(bot_p):
        pass
        # if bot_p[n_b]

    return line

def plot_trajectory(agent_p, bot_p, n_agents, n_bots):
    numDataPoints = len(agent_p)

    fig = plt.figure()
    ax = Axes3D(fig)
    
    color_list = ['g', 'b', 'r', 'y', 'o']
    
    line = [None for _ in range(n_agents)]
    redDots = [None for _ in range(n_bots)]
    agentDots = [None for _ in range(n_bots)]
    xdata = [[] for _ in range(n_agents)]
    ydata = [[] for _ in range(n_agents)]
    zdata = [[] for _ in range(n_agents)]

    for n_b in range(n_bots):
        redDots[n_b] = ax.scatter3D(bot_p[0][n_b][0], bot_p[0][n_b][1], bot_p[0][n_b][2], lw=14, c='r')

    for n_a in range(n_agents):
        agentDots[n_a] = ax.scatter3D(agent_p[0][n_a][0], agent_p[0][n_a][1], agent_p[0][n_a][2], lw=4, c=color_list[n_a])

    for point_idx in range(len(agent_p)):
        for n_a in range(n_agents):
            xdata[n_a].append(agent_p[point_idx][n_a][0])
            ydata[n_a].append(agent_p[point_idx][n_a][1])
            zdata[n_a].append(agent_p[point_idx][n_a][2])

    for n_a in range(n_agents):
        line[n_a] = plt.plot(xdata[n_a], ydata[n_a], zdata[n_a], lw=2, c=color_list[n_a])[0]
        
    ax.set_xlim((-31, 31))
    ax.set_ylim((-31, 31))
    ax.set_zlim((-15, 15))

    ax.set_xlabel('X - Axis')
    ax.set_ylabel('Y - Axis')
    ax.set_zlabel('Z - Axis')
    ax.set_title('Trajectory of agents')
    
    line_ani = animation.FuncAnimation(fig, animate, frames=numDataPoints, fargs=(xdata, ydata, zdata, line, n_agents, bot_p, redDots), interval=50, blit=False)

    plt.show()
    
if __name__ == "__main__":
    with open('./agents_position/agents_positions.pkl', 'rb') as f:
        agents_pos_list = pickle.load(f)

    with open('./agents_position/bots_positions.pkl', 'rb') as f:
        bots_pos_list = pickle.load(f)

    n_agents = len(agents_pos_list[0][0])
    n_bots = len(bots_pos_list[0][0])

    for idx in range(len(agents_pos_list)):
        agent_pos, bot_pos = agents_pos_list[idx], bots_pos_list[idx]
        plot_trajectory(agent_pos, bot_pos, n_agents, n_bots)