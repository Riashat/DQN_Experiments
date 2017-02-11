import copy
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time


from utils import *
from ReplayMemory import ReplayMemory
from agents import AgentEpsGreedy
from valuefunctions import ValueFunctionDQN

# Inspired by necnec's algorithm at:
#   https://gym.openai.com/evaluations/eval_89nQ59Y4SbmrlQ0P9pufiA
# And inspired by David Silver's Deep RL tutorial:
#   http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf

# results_dir_prefix = '###'
# upload = False

discount = 0.99
decay_eps = 0.9
batch_size = 64
max_n_ep = 1000

#USING A LARGE CUMULATIVE MINIMUM AVERAGE REWARD FOR THE CART POLE HERE
min_avg_Rwd = 2000  # Minimum average reward to consider the problem as solved
n_avg_ep = 100      # Number of consecutive episodes to calculate the average reward

# t = get_last_folder_id(results_dir_prefix) + 1  # Calculate next test id
# results_dir = results_dir_prefix + '\\' + str(t).zfill(4)
# os.makedirs(results_dir)


def run_episode(env,
                agent,
                state_normalizer,
                memory,
                batch_size,
                discount,
                max_step=10000):
    state = env.reset()
    if state_normalizer is not None:
        state = state_normalizer.transform(state)[0]
    done = False
    total_reward = 0
    step_durations_s = np.zeros(shape=max_step, dtype=float)
    train_duration_s = np.zeros(shape=max_step-batch_size, dtype=float)
    progress_msg = "Step {:5d}/{:5d}. Avg step duration: {:3.1f} ms. Avg train duration: {:3.1f} ms. Loss = {:2.10f}."
    loss_v = 0
    w1_m = 0
    w2_m = 0
    w3_m = 0
    i = 0
    action = 0
    for i in range(max_step):
        t = time.time()
        if i > 0 and i % 200 == 0:
            print(progress_msg.format(i, max_step,
                                      np.mean(step_durations_s[0:i])*1000,
                                      np.mean(train_duration_s[0:i-batch_size])*1000,
                                      loss_v))
        if done:
            break
        action = agent.act(state)

        state_next, reward, done, info = env.step(action)
        total_reward += reward

        if state_normalizer is not None:
            state_next = state_normalizer.transform(state_next)[0]
        memory.add((state, action, reward, state_next, done))

        if len(memory.memory) > batch_size:  # DQN Experience Replay
            states_b, actions_b, rewards_b, states_n_b, done_b = zip(*memory.sample(batch_size))
            states_b = np.array(states_b)
            actions_b = np.array(actions_b)
            rewards_b = np.array(rewards_b)
            states_n_b = np.array(states_n_b)
            done_b = np.array(done_b).astype(int)
            q_n_b = agent.predict_q_values(states_n_b)  # Action values on the arriving state
            targets_b = rewards_b + (1. - done_b) * discount * np.amax(q_n_b, axis=1)

            targets = agent.predict_q_values(states_b)
            for j, action in enumerate(actions_b):
                targets[j, action] = targets_b[j]

            t_train = time.time()
            loss_v, w1_m, w2_m, w3_m = agent.train(states_b, targets)
            train_duration_s[i - batch_size] = time.time() - t_train

        state = copy.copy(state_next)
        step_durations_s[i] = time.time() - t  # Time elapsed during this step
    return loss_v, w1_m, w2_m, w3_m, total_reward

env = gym.make("CartPole-v0")

n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]

value_function = ValueFunctionDQN(state_dim=state_dim, n_actions=n_actions, batch_size=batch_size)
agent = AgentEpsGreedy(n_actions=n_actions, value_function_model=value_function, eps=0.9)
memory = ReplayMemory(max_size=100000)


Experiments = 3
Experiments_All_Rewards = np.zeros(shape=(max_n_ep))


for e in range(Experiments):

    print('Experiment Number ', e)


    loss_per_ep = []
    w1_m_per_ep = []
    w2_m_per_ep = []
    w3_m_per_ep = []
    total_reward = []

    ep = 0
    avg_Rwd = -np.inf
    episode_end_msg = 'loss={:2.10f}, w1_m={:3.1f}, w2_m={:3.1f}, w3_m={:3.1f}, total reward={}'

    while avg_Rwd < min_avg_Rwd and ep < max_n_ep:
        if ep >= n_avg_ep:
            avg_Rwd = np.mean(total_reward[ep-n_avg_ep:ep])
            print("EPISODE {}. Average reward over the last {} episodes: {}.".format(ep, n_avg_ep, avg_Rwd))
        else:
            print("EPISODE {}.".format(ep))


        loss_v, w1_m, w2_m, w3_m, cum_R = run_episode(env, agent, None, memory, batch_size=batch_size, discount=discount,
                                                      max_step=15000)
        print(episode_end_msg.format(loss_v, w1_m, w2_m, w3_m, cum_R))

        if agent.eps > 0.0001:
            agent.eps *= decay_eps

        # Collect episode results
        loss_per_ep.append(loss_v)
        w1_m_per_ep.append(w1_m)
        w2_m_per_ep.append(w2_m)
        w3_m_per_ep.append(w3_m)
        total_reward.append(cum_R)

        ep += 1

    Experiments_All_Rewards = Experiments_All_Rewards + total_reward

    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments/Average_DQN_CartPole_V0_Results/'  + 'CartPole_V0_cumulative_reward' + 'Experiment_' + str(e) + '.npy', total_reward)
    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments/Average_DQN_CartPole_V0_Results/'  + 'CartPole_V0_value_function_loss' + 'Experiment_' + str(e) + '.npy', loss_per_ep)


env.close()



print('Saving Average Cumulative Rewards Over Experiments')

Average_Cum_Rwd = np.divide(Experiments_All_Rewards, Experiments)

np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments/Average_DQN_CartPole_V0_Results/'  + 'Average_Cum_Rwd_CartPole_V0' + '.npy', Average_Cum_Rwd)


print "All Experiments DONE"



#####################
#PLOT RESULTS
eps = range(ep)
plt.figure()
plt.subplot(211)
plt.plot(eps, Average_Cum_Rwd)
Rwd_avg = movingaverage(Average_Cum_Rwd, 100)
plt.plot(eps[len(eps) - len(Rwd_avg):], Rwd_avg)
plt.xlabel("Episode number")
plt.ylabel("Reward per episode")
plt.grid(True)
plt.title("Total reward")

