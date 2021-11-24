from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import pickle


class AnimatedVoxels(object):
    def __init__(self, nagents, poslist_filename="./agents_position/agents_positions_planner.pkl"):
        with open(poslist_filename, 'rb') as f:
            self.position_list = pickle.load(f)
        self.sim_len = np.asarray(self.position_list).shape[1]
        
        self.nagents = nagents
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ani = animation.FuncAnimation(self.fig, self.update, self.sim_len, interval=20, repeat=False)

        x = np.arange(0, 40, 1)
        y = np.arange(0, 40, 1)
        z = np.arange(0, 6, 2)
        self.x, self.y, self.z = np.meshgrid(x, y, z)
        self.ax.set_aspect('auto')

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")


        self.color_name = 'red'
        plt.xlim(0, 40)
        plt.ylim(0, 40)
        plt.show()


    def update(self, num):
        voxel_list = []
        for i in range(self.nagents):
            voxel_list.append((self.x == self.position_list[0][num][i][0]+20) & (self.y == self.position_list[0][num][i][1]+20)& (self.z == self.position_list[0][num][i][2]))
        
        voxels = voxel_list[0]

        for i in range(self.nagents-1):
            voxels = voxels | voxel_list[i+1]

        #voxels = voxel_list[0] | voxel_list[1] | voxel_list[2] | voxel_list[3] | voxel_list[4]
        colors = np.empty(voxels.shape, dtype=object)
        for i in range(self.nagents):
            colors[voxel_list[i]] = self.color_name
            #voxel_list[i] = plt.plot(xdata[n_a], ydata[n_a], zdata[n_a], lw=2, c=color_list[n_a])[0]
            #voxel_list[i].set_label('Agent %d' %(i+1))

        self.ax.voxels(voxels, facecolors=colors, edgecolor='k',label="drone")

if __name__ == "__main__":
    AnimatedVoxels(5)
