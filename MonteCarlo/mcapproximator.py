import numpy as np
import random
import networkx as nx
from utils import *


class MCAgent_approx:

    def __init__(self, env, gamma = 0.9, min_epsilon = 0.7, e_decay_exponentially = True,
        max_epsilon = 1, n_episodes = 1000, max_steps = 100, feature_type = "one_hot", seed = None):
        self.env = env
        self.G = env.G
        # self.Q = {}
        self.n_states = env.get_n_states()
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon = max_epsilon
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.seed = seed
        self.feature_type = feature_type
        if seed is not None:
            self.set_seed(seed)

        self.features = np.zeros((self.n_states, 3))
        for state in range(self.n_states):
            self.features[state, 0] = self.G.nodes[state]["x"]
            self.features[state, 1] = self.G.nodes[state]["y"]
            self.features[state, 2] = len(self.G[state])
        self.features[:, 0] = self.features[:, 0] / np.max(self.features[:, 0])
        self.features[:, 1] = self.features[:, 1] / np.max(self.features[:, 1])
        self.features[:, 2] = self.features[:, 2] / np.max(self.features[:, 2])

        self.input_dim = self.get_feature(0, 0).shape[0]
        self.weights = np.zeros(shape=self.input_dim)
        self.set_N()
        self.e_decay_exponentially = e_decay_exponentially

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_feature(self, state, action):
        if self.feature_type == "position":
            features = np.concatenate((self.features[state], self.features[action]))
        elif self.feature_type == "one_hot":
            features = np.zeros(self.n_states * 2 + 6)
            features[state] = 1
            features[self.n_states + action] = 1
            features[-6:-3] = self.features[state]
            features[-3:] = self.features[action]
        elif self.feature_type == "position_large":
            features = np.zeros(self.n_states * 3)
            features[3 * state : 3 * state + 3] = self.features[state]
        return features

    def linear_func(self, state, action, return_feature = False):
        """For each pair of state and action, compute 6 features:
        (x, y) coordinates of the state/action, and the number of neighbors of each one.
        Then, compute the linear combination of the features with the weights and bias.

        Parameters
        ----------
        state : int
            Index of the state
        action : int
            Index of the action

        Returns
        -------
        float
            Q value of the pair (state, action)
        """
        features = self.get_feature(state, action)
        prediction = np.dot(self.weights, features)
        if return_feature:
            return prediction, features
        return prediction

    def set_N(self):
        self.N = np.full((len(self.env.G.nodes), len(self.env.G.nodes)), -np.inf)

        for i in self.env.G.nodes:
            neighbors = list(self.env.G.neighbors(i))
            self.N[i, neighbors] = 0

    def update_epsilon(self, s=''):
        # make it decay exponentially
        self.epsilon -= (self.max_epsilon - self.min_epsilon) / self.n_episodes

    def step(self, state):
        """Returns the next action epsilon greedily using value function."""
        n_random = np.random.uniform(0, 1)

        if self.e_decay_exponentially is False:
            n = np.sum(self.N[state, :][self.N[state, :] != -np.inf])
            self.epsilon = self.min_epsilon / (self.min_epsilon + n)

        neighbors = list(self.env.G.neighbors(state))
        if len(neighbors) == 0:
            return int(state)

        if n_random < self.epsilon:  # explore

            return random.choice(neighbors)
        else:  # exploit
            return self.greedy_policy(state)

    def generate_episode(self, observation):
        """Returns the list with state, action and rewards of the episode"""
        # it initializes list of tuples (state, action and reward)
        episode = []
        state = observation

        for step in range(self.max_steps):

            action = self.step(state)  # select an action
            new_state, reward, done = self.env.step(action)  # take action

            episode.append((state, action, reward))
            state = new_state

            if done is True:  # if end simulation, break
                self.routes = []
                route = [observation] + [i[1] for i in episode]
                if route[-1] != self.env.target:
                    self.routes.append(route)
                break

        self.evaluation(episode)

        return episode

    def evaluation(self, episode):
        G = 0
        w = 0
        for i in range(len(episode) - 2, -1, -1):

            # state, action, reward
            s, a, r = episode[i]
            _, _, r_t1 = episode[i + 1]

            find_sa_pair = len(list(filter(lambda step: step[0] == s and step[1] == a, episode[:i]))) > 0

            # check if s,a are the first visit
            if find_sa_pair is False:
                # G <- gamma * G + R(t+1)
                G = (self.gamma * G) + r_t1
                self.N[s, a] += 1
                # print('features', features)

                prediction, features = self.linear_func(s, a, True)

                # w = w + alpha [(G - x(s)^T * w) * x(s)]
                for i in range(len(self.weights)):

                    self.weights[i] += (1 / self.N[s, a]) * ((G - prediction) * features[i])
                    #self.weights[i] += 0.6 * ((G - prediction) * features[i])
                    # print(G, prediction, features[i], self.weights[i], s, a)
                # print('w', self.weights)

            # print('weights', self.weights)
        return

    def argmax(self, state):
        neighbors = list(self.G.neighbors(state)) + [state]
        return np.max([self.linear_func(state, action) for action in neighbors])

    def greedy_policy(self, state):
        """Greedy policy that returns the action with the highest Q value"""
        neighbors = list(self.G.neighbors(state)) + [state]
        return neighbors[np.argmax([self.linear_func(state, action) for action in neighbors])]


    def update_weights(self, reward, state, action, new_state):
        """Update the weights and bias based on the Bellman equation"""
        target = reward + self.gamma * self.argmax(new_state)
        prediction, features = self.linear_func(state, action, True)
        error = target - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * features[i]

    def train(self):
        self.epsilon = self.max_epsilon

        self.episode_rewards = []
        self.returns = {}
        observation = self.env.reset()
        for _ in range(self.n_episodes):

            episode = self.generate_episode(observation)

            self.episode_rewards.append(np.sum([i[2] for i in episode]))

            if self.e_decay_exponentially is True:
                self.update_epsilon()

            observation = self.env.reset_linear()

        self.policy = {i : self.greedy_policy(i) for i in range(self.n_states)}
        print(self.policy)
    # def route_to_target(self, source, target):
    #     route = [source]
    #     state = source
    #     cost = 0
    #     k = 0
    #     while state != target and k < 1000:
    #         new_state = self.policy[state]
    #         if new_state == state:
    #             cost = np.inf
    #             route.append(new_state)
    #             break
    #         cost += self.env.G[state][new_state][0]["length"]
    #         state = new_state
    #         route.append(state)
    #         k += 1
    #     return route
    #
    # def route_cost(self, env):
    #     source = env.source
    #     target = env.target
    #     route = [source]
    #     state = source
    #     cost = 0
    #     k = 0
    #     while state != target and k < 1000:
    #         new_state = self.policy[state]
    #         if new_state == state:
    #             cost = np.inf
    #             route.append(new_state)
    #             break
    #         cost += env.G[state][new_state][0]["length"]
    #         state = new_state
    #         route.append(state)
    #         k += 1
    #     return cost

    def route_to_target(self, source, target):
        route = [source]
        state = source
        k = 0
        while state != target and k < 1000:

            state = self.policy[state]
            route.append(state)
            k += 1
        return route

    def route_cost(self, env):
        route = self.route_to_target(env.source, env.target)
        try:
            cost = float(nx.path_weight(env.G, route, "length"))

        except:
            cost = np.inf

        return cost
