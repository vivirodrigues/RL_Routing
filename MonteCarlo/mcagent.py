import osmnx as ox;
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import scipy.stats
from utils import *
from time import time


class MCAgent:
    def __init__(
            self,
            env,
            gamma=0.99,
            n_episodes=1000,
            max_steps=1000,
            min_epsilon = 0.1,
            seed=42
    ):
        self.env = env
        self.Q = {}
        # self.set_Q()
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.max_epsilon = 1
        self.min_epsilon = min_epsilon
        self.epsilon = self.max_epsilon
        self.seed = seed
        self.set_seed()
        self.randomicos = []
        self.policy = {}
        self.routes = []

    # def set_Q(self):
    #     self.Q = np.full((len(self.env.G.nodes), len(self.env.G.nodes)), -np.inf)
        
    #     for i in self.env.G.nodes:
    #         neighbors = list(self.env.G.neighbors(i))        
    #         self.Q[i, neighbors] = 0                            
    
    
    def update_epsilon(self):
        # make it decay exponentially
        self.epsilon -= (self.max_epsilon - self.min_epsilon) / self.n_episodes

    # def greedy_policy(self, state):
    #     """Greedy policy that returns the action with the highest Q value"""    
    #     return np.random.choice(np.where(self.Q[state, :] == np.max(self.Q[state, :]))[0])

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def step(self, state):
        """Returns the next action epsilon greedily using value function."""
        n_random = np.random.uniform(0, 1)
        self.randomicos.append(n_random)

        if n_random < self.epsilon:  # explore
            neighbors = list(self.env.G.neighbors(state))  # + [state]
            return random.choice(neighbors)
        else:  # exploit            
            # return self.greedy_policy(state)
            action, got_it = argmax(self.Q, state)
            if got_it is False:  # if not possible to chose the best action, chose randomly
                neighbors = list(self.env.G.neighbors(state))  # + [state]
                return random.choice(neighbors)
            else:
                return int(action)

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
        
        for i in range(len(episode) - 2, -1, -1):

            # state, action, reward
            s, a, r = episode[i]
            _, _, r_t1 = episode[i + 1]

            # G <- gamma * G + R(t+1)
            G = (self.gamma * G) + r_t1

            find_sa_pair = len(list(filter(lambda step: step[0] == s and step[1] == a, episode[:i]))) > 0

            # check if s,a are the first visit
            if find_sa_pair is False:
                r_ = self.returns.get((s, a))
                if r_ is None:
                    self.returns.update([((s, a), [G])])
                else:
                    r_.append(G)
                    self.returns.update([((s, a), r_)])
        
        for i, j in list(self.returns.items()):
            valor_medio = np.sum(j) / len(j)
            self.Q.update([((i[0], i[1]), valor_medio)])
            # self.Q[i[0], i[1]] = valor_medio
        
        
        # print(np.argmax(np.array([tempo1, tempo2])))
        # print(tempo1, tempo2)
        
        return

    def train(self):        
        self.epsilon = self.max_epsilon
        
        self.episode_rewards = []
        self.returns = {}
        for _ in range(self.n_episodes):
            
            observation = self.env.reset()            
            
            episode = self.generate_episode(observation)           
            
            self.episode_rewards.append(np.sum([i[2] for i in episode]))            
            
            self.update_epsilon()                       

        self.policy = {i: argmax(self.Q, i)[0] for i in range(len(self.env.G.nodes))}
        # self.policy = {i: self.greedy_policy(i) for i in range(len(self.env.G.nodes))}

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