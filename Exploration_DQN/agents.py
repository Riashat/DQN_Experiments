import numpy as np


class AgentEpsGreedy:
    def __init__(self, n_actions, value_function_model, eps=1.0):
        self.n_actions = n_actions
        self.value_func = value_function_model
        self.eps = eps

    def act(self, state):
        action_values = self.value_func.predict([state])[0]

        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(action_values)
        policy[a_max] += 1. - self.eps

        return np.random.choice(self.n_actions, p=policy)

    def train(self, states, targets):
        return self.value_func.train(states, targets)

    def predict_q_values(self, states):
        return self.value_func.predict(states)


    def act_boltzmann(self, state):
        action_values = self.value_func.predict([state])[0]
        action_values_tau = action_values / self.eps
        policy = np.exp(action_values_tau) / np.sum(np.exp(action_values_tau), axis=0)

        action_value_to_take = np.argmax(policy)
        
        return action_value_to_take


    def act_random(self, state):
        action = np.random.randint(0, self.n_actions)

        return action



    def act_mc_dropout_Boltzmann(self, state):

        dropout_iterations = 5
        dropout_acton_values = np.array([[0, 0]]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict([state])[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

        dropout_acton_values = dropout_acton_values[:, 1:]

        mean_action_values = np.mean(dropout_acton_values, axis=1)

        Q_mean = mean_action_values 

        Q_dist = np.exp(Q_mean) / np.sum(np.exp(Q_mean), axis=0)

        # Q_dist[a_max] += 1. - self.eps

        actions_to_take = np.argmax(Q_dist)

        return actions_to_take



    def act_mc_dropout_EpsilonGreedy(self, state):

        dropout_iterations = 5
        dropout_acton_values = np.array([[0, 0]]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict([state])[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

        dropout_acton_values = dropout_acton_values[:, 1:]

        mean_action_values = np.mean(dropout_acton_values, axis=1)

        policy = np.ones(self.n_actions) * self.eps / self.n_actions

        a_max = np.argmax(mean_action_values)

        policy[a_max] += 1. - self.eps

        action_to_take = np.random.choice(self.n_actions, p=policy)

        return action_to_take



