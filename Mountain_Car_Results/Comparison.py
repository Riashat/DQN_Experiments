import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from plotting import *


eps = 900
eps = range(eps)


mountain_car_DQN = np.load('Average_Cum_Rwd_Mountain_Car.npy')

linear_mountain_car = np.load('linear_approx_mountain_car_cumulative_reward.npy')

mountain_car_DQN_Exp0 = np.load('mountain_car_cumulative_rewardExperiment_0.npy')
mountain_car_DQN_Exp1 = np.load('mountain_car_cumulative_rewardExperiment_1.npy')
mountain_car_DQN_Exp2 = np.load('mountain_car_cumulative_rewardExperiment_2.npy')


# plt.figure(1)
# plt.plot(eps, mountain_car_DQN, 'red')
# plt.xlabel("Episode number (Last Few Episodes")
# plt.ylabel("Cumulative Reward per Episode")
# plt.grid(True)
# plt.title("DQN - Mountain Car")
# plt.show()



# plt.figure(2)
# plt.plot(eps, mountain_car_DQN_Exp1, 'blue')
# plt.xlabel("Episode number (Last Few Episodes")
# plt.ylabel("Cumulative Reward per Episode")
# plt.grid(True)
# plt.title("DQN - Mountain Car")
# plt.show()


def plot_episode_stats(stats1, stats2, eps,  smoothing_window=50, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Linear Function Approximator on Mountain Car")
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Deep Q Learning on Mountain Car")
    plt.legend(handles=[cum_rwd_1, cum_rwd_2])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Linear and Non-Linear Function Approximators on Mountain Car")
    plt.show()


    return fig



def main():
	plot_episode_stats(linear_mountain_car, mountain_car_DQN, eps)


if __name__ == '__main__':
	main()



