import numpy as np
import pickle
import os
import glob
import csv
import pandas as pd
import gc
gc.enable()
##########################################


class Reporter:
    def __init__(self):
        self.stats_columns = ['model', 'coverage mean', 'coverage std']
        self.stats_filename = 'training_results.csv'

        x_lim = 20  # grid x limit
        y_lim = 20  # grid y limit
        z_lim = 6  # grid z limit
        res = 1.0  # resolution for grids
        out_shape = 164  # width and height for uncertainty matrix
        dist = 5.0  # distance threshold

        X, Y, Z = np.mgrid[-x_lim:x_lim + 0.1:res,
                           -y_lim:y_lim + 0.1:res,
                           0:z_lim + 0.1: 2 * res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T
        self.uncertainty_values = np.random.uniform(
            low=0.95, high=1.0, size=(self.uncertainty_grids.shape[0],))

    def get_closest_n_grids(self, current_pos, n):
        print("Currrent POS:", current_pos)
        print("uncertainty_grids:", self.uncertainty_grids)
        differences = current_pos - self.uncertainty_grids
        distances = np.sum(differences * differences, axis=1)
        sorted_indices = sorted(range(len(distances)),
                                key=lambda k: distances[k])
        return sorted_indices[0:n]

    def get_map_coverage(self, pkl):
        average_val_list = []

        with open('./agents_position/' + pkl, 'rb') as f:
            total_pos_list = pickle.load(f)
            #print("Total pos list: ", np.array(total_pos_list).shape)
        for episode in range(len(total_pos_list)):
            grid_visits = np.zeros((self.uncertainty_grids.shape[0], ))
            episode_pos_list = np.array(total_pos_list[episode])
            for iteration in range(episode_pos_list.shape[0]):
                for agent_ind in range(episode_pos_list.shape[1]):
                    agent_pos = episode_pos_list[iteration, agent_ind, 0:3]
                    indices = self.get_closest_n_grids(agent_pos, 8)
                    grid_visits[indices] = 1

            visited_grids = len(np.where(grid_visits == 1)[0])
            average_val = 100 * visited_grids / len(grid_visits)
            average_val_list.append(average_val)
            print("Episode: {0} Map coverage: {1:.4}".format(
                episode, average_val))

        print("=====================================================")
        print("=====================================================")
        print("Average map coverage: ", np.mean(average_val_list))
        print("Std map coverage: ", np.std(average_val_list))
        print("=====================================================")
        print("=====================================================")

        self.write_stats(
            [str(pkl), np.mean(average_val_list), np.std(average_val_list)])
        gc.collect()

    def write_stats(self, stats):
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))
