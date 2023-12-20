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
        elif self.feature_type == "x":
            features = np.zeros(2)
            features[0] = self.env.G.nodes[state]['x']
            features[1] = self.env.G.nodes[state]['y']
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
        #print(prediction)

        prediction = np.dot(self.weights, features)
        print(prediction)
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
                
                self.N[s, a] += 1

                prediction, features = self.linear_func(s, a, True)

                
                for i in range(len(self.weights)):
                    G += r_t1
                    #G = (self.gamma * G) + r_t1

                    # w = w + alpha [(G - x(s)^T * w) * x(s)]
                    self.weights[i] += 0.5 * ((G - prediction) * features[i])
                    

            # print('weights', self.weights)
        return

    def argmax(self, state):
        neighbors = list(self.G.neighbors(state)) + [state]
        return np.max([self.linear_func(state, action) for action in neighbors])

    def greedy_policy(self, state):
        """Greedy policy that returns the action with the highest Q value"""
        neighbors = list(self.G.neighbors(state)) + [state]
        return neighbors[np.argmax([self.linear_func(state, action) for action in neighbors])]

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
