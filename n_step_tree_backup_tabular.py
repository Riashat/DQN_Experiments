import gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random

from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict
from lib.envs.gridworld import GridworldEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
	

env = GridworldEnv()



def make_epsilon_greedy_policy(Q, epsilon, nA):

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn

def chosen_action(Q):
	best_action = np.argmax(Q)
	return best_action


def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn





def q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy
	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  
	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
	
	for i_episode in range(num_episodes):
		state = env.reset()
		for t in itertools.count():
			#take a step in the environmnet
			#choose action A using policy derived from Q
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			
			# with taken aciton, observe the reward and the next state
			next_state, reward, done, _, = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# TD Update Equations:
			# max_a of Q(s', a) - where s' is the next state, and we consider all maximising over actions which was derived 
			#from previous policy based on Q
			best_next_action = np.argmax(Q[next_state])
			td_target = reward + discount_factor * Q[next_state][best_next_action]
			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state

	return Q, stats



"""
Expected SARSA Algorithm = 1 step TREE BACKUP
"""

def one_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

	#Expected SARSA : same algorithm steps as Q-Learning, 
	# only difference : instead of maximum over next state and action pairs
	# use the expected value
	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		state = env.reset()

		#steps within each episode
		for t in itertools.count():
			#pick the first action
			#choose A from S using policy derived from Q (epsilon-greedy)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			#reward and next state based on the action chosen according to epislon greedy policy
			next_state, reward, done, _ = env.step(action)
			
			#reward by taking action under the policy pi
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#pick the next action
			# we want an expectation over the next actions 
			#take into account how likely each action is under the current policy

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )


			#V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
			V = np.sum(next_action_probs * Q[next_state])

			#Update rule in Expected SARSA
			td_target = reward + discount_factor * V
			td_delta = td_target - Q[state][action]

			Q[state][action] += alpha * td_delta


			if done:
				break
			state = next_state

	return Q, stats



def two_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):

		state = env.reset()

		#steps within each episode
		for t in itertools.count():
			#pick the first action
			#choose A from S using policy derived from Q (epsilon-greedy)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			#reward and next state based on the action chosen according to epislon greedy policy
			next_state, reward, _ , _ = env.step(action)
			
			#reward by taking action under the policy pi
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#pick the next action
			# we want an expectation over the next actions 
			#take into account how likely each action is under the current policy

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )


			#V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
			V = np.sum(next_action_probs * Q[next_state])


			next_next_state, next_reward, done, _ = env.step(next_action)

			# stats.episode_rewards[i_episode] += next_reward
			# stats.episode_lengths[i_episode] = t
	
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			next_V = np.sum(next_next_action_probs * Q[next_next_state])			


			Delta = next_reward + discount_factor * next_V - Q[next_state][next_action]

			td_target = reward + discount_factor * V +  discount_factor *  next_action * Delta

			td_delta = td_target - Q[state][action]


			Q[state][action] += alpha * td_delta


			if done:
				break

			state = next_state

	return Q, stats



def three_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

	#Expected SARSA : same algorithm steps as Q-Learning, 
	# only difference : instead of maximum over next state and action pairs
	# use the expected value
	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		state = env.reset()

		#steps within each episode
		for t in itertools.count():
			#pick the first action
			#choose A from S using policy derived from Q (epsilon-greedy)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			#reward and next state based on the action chosen according to epislon greedy policy
			next_state, reward, _ , _ = env.step(action)
			
			#reward by taking action under the policy pi
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )


			#V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
			V = np.sum(next_action_probs * Q[next_state])


			next_next_state, next_reward, _, _ = env.step(next_action)
	
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			next_V = np.sum(next_next_action_probs * Q[next_next_state])			


			Delta = next_reward + discount_factor * next_V - Q[next_state][next_action]

			two_step_target = reward + discount_factor * V +  discount_factor *  next_action * Delta


			next_next_next_state, next_next_reward, done, _ = env.step(next_next_action)
			next_next_next_action_probs	 = policy(next_next_next_state)
			next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

			next_next_V = np.sum(next_next_next_action_probs * Q[next_next_next_state])

			next_Delta = next_next_reward + discount_factor * next_next_V - Q[next_next_state][next_next_action]

					
			td_target = two_step_target + discount_factor * next_action * discount_factor * next_next_action * next_Delta




			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta


			if done:
				break

			state = next_state

	return Q, stats









def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	action_space = env.action_space.n

	#on-policy which the agent follows - we want to optimize this policy function
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	#each step in the episode
	for i_episode in range(num_episodes):

		state = env.reset()
		action_probs = policy(state)

		#choose a from policy derived from Q (which is epsilon-greedy)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		#for every one step in the environment
		for t in itertools.count():
			#take a step in the environment
			# take action a, observe r and the next state
			next_state, reward, done, _ = env.step(action)

			#choose a' from s' using policy derived from Q
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(action_probs)), p = next_action_probs)

			#update cumulative count of rewards based on action take (not next_action) using Q (epsilon-greedy)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# TD Update Equations
			#TD Target - One step ahead
			td_target = reward + discount_factor * Q[next_state][next_action]
			
			# TD Error
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta
			if done:
				break
			action = next_action
			state = next_state


	return Q, stats







def plot_episode_stats(stats1, stats2, stats3, stats4, stats5, smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="SARSA")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="One Step Tree Backup")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Two Step Tree Backup")
    cum_rwd_4, = plt.plot(rewards_smoothed_4, label="Three Step Tree Backup")
    cum_rwd_5, = plt.plot(rewards_smoothed_5, label="Q Learning")

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("n-step Tree Backup, SARSA and Q-Learning")
    plt.show()


    return fig


def trial_plot(stats, smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(rewards_smoothed, label="Two Step Tree Backup")

    plt.legend(handles=[cum_rwd])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Trial Plot")
    plt.show()


    return fig





def main():

	Number_Episodes = 1500

	print "SARSA"
	SARSA, stats_SARSA = sarsa(env, Number_Episodes)
	
	print "One Step Tree Backup"
	Expected_SARSA, stats_expected_sarsa = one_step_tree_backup(env, Number_Episodes)


	print "Two Step Tree Backup"
	two_step_tree_backup_Q, stats_two_step_tree_backup = two_step_tree_backup(env, Number_Episodes)


	print "Three Step Tree Backup"
	three_step_tree_backup_Q, stats_three_step_tree_backup = three_step_tree_backup(env, Number_Episodes)
	
	print "Q-Learning"
	Q_learning_Q, stats_q_learning = q_learning(env, Number_Episodes)
	

	print "Plotting Cumulative Rewards"
	plot_episode_stats(stats_SARSA, stats_expected_sarsa, stats_two_step_tree_backup, stats_three_step_tree_backup, stats_q_learning)


if __name__ == '__main__':
	main()



