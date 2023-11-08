import numpy as np


class Environment:
    """
    Simulation of graph route that can be deterministic and stochastic.
    The main functionalities is to reset the simulation and returns the state based in the action.
    """

    def __init__(self, G, target, reward="unit", mode="deterministic"):
        self.G = G
        self.target = target
        self.reward = reward
        if mode == "deterministic":
            self.step = self.step_deterministic
        elif mode == "stochastic":
            self.step = self.step_stochastic

    def get_n_states(self):
        return len(self.G.nodes)

    def reset(self):
        """Return a random state that is not the destination"""
        self.state = np.random.choice(list(self.G.nodes))
        while self.state == self.target:
            self.state = np.random.choice(list(self.G.nodes))
        return self.state

    def step_deterministic(self, action):
        """Return new state, reward, and if the destination is reached"""
        neighbors = list(self.G.neighbors(self.state))
        if action not in neighbors:  # action is not possible
            return self.state, -10

        w = self.G[self.state][action]["weight"]
        self.state = action
        if self.state == self.target:
            return self.state, 10
        else:
            if self.reward == "unit":
                return self.state, 0
            elif self.reward == "weighted":
                return self.state, -w

    def step_stochastic(self, action):
        """Return new state, reward, and if the destination is reached"""
        if np.random.rand() < 0.05:  # 5% probability of not moving
            return self.state, 0

        neighbors = list(self.G.neighbors(self.state))
        if action not in neighbors:  # action is not possible
            return self.state, -10

        w = self.G[self.state][action]["weight"]
        self.state = action
        if self.state == self.target:
            return self.state, 10
        else:
            if self.reward == "unit":
                return self.state, 0
            elif self.reward == "weighted":
                return self.state, -(w + np.random.normal(scale=0.1))

    def is_finished(self):
        return self.state == self.target
