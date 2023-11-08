import numpy as np


class Q_learning:
    def __init__(
        self,
        env,
        learning_rate=0.7,
        gamma=0.95,
        min_epsilon=0.05,
        max_epsilon=1,
        n_episodes=1000,
        max_steps=100,
    ):
        """Training of the Q table using Q learning algorithm.
        It needs the enviroment env.

        :param Q: numpy array with the Q table
        :param learning_rate: learning rate of algorithm, defaults to 0.7
        :param gamma: weight of future rewards, defaults to 0.95
        :param min_epsilon: min probability of exploration, defaults to 0.05
        :param max_epsilon: max probability of exploration, defaults to 1
        :param n_episodes: number of episodes, defaults to 1000
        :param max_steps: max number of steps per episode, defaults to 100
        """
        self.env = env
        self.n_states = env.get_n_states()
        self.Q = np.zeros((self.n_states, self.n_states))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.n_episodes = n_episodes
        self.max_steps = max_steps

    def greedy_policy(self, state):
        """Greedy policy that returns the action with the highest Q value"""
        return np.argmax(self.Q[state, :])

    def epsilon_greedy_policy(self, state, epsilon):
        """Epsilon greedy policy that returns a random action with probability epsilon"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.n_states)
        else:
            return self.greedy_policy(state)

    def train(self):
        epsilon = self.max_epsilon
        self.episode_rewards = []
        self.reached_target = np.inf
        
        for episode in range(self.n_episodes):
            # Start episode again resetting the enviroment
            state = self.env.reset()
            self.episode_rewards.append(0)

            for step in range(self.max_steps):
                # Choose action and get reward
                action = self.epsilon_greedy_policy(state, epsilon)
                new_state, reward = self.env.step(action)
                self.episode_rewards[-1] += reward    

                # update Q table based on Bellman equation
                self.Q[state, action] += self.learning_rate * (
                    reward
                    + self.gamma * np.max(self.Q[new_state, :])
                    - self.Q[state, action]
                )
                state = new_state

                if self.env.is_finished():
                    self.reached_target = min(self.reached_target, episode)
                    break

            epsilon -= (
                self.max_epsilon - self.min_epsilon
            ) / self.n_episodes  # make it decay exponentially

    def evaluate(self, source=None):
        if source is None:
            state = self.env.reset()
        else:
            state = source
            self.env.state = source
        actions = [state]
        for _ in range(100):
            action = self.greedy_policy(state)
            state, _ = self.env.step(action)
            actions.append(action)
            if self.env.is_finished():
                break
        if actions[-1] != self.env.target:
            actions = []
        return actions
    
    def run(self, source=None):
        if source is None:
            state = self.env.reset()
        else:
            state = source
            self.env.state = source
        
        actions = [state]
        for _ in range(100):
            action = self.greedy_policy(state)
            state, _ = self.env.step(action)
            actions.append(action)
            if self.env.is_finished():
                break
        return actions

    def evaluate_agg(self, source = None, n_trials = 10):
        """It will try to find the best path from source to target n_trials times."""
        trials_results = []
        for _ in range(n_trials):
            if source is None:
                state = self.env.reset()
            else:
                state = source
                self.env.state = source
            
            reward = 0
            for _ in range(100):
                action = self.greedy_policy(state)
                state, r = self.env.step(action)
                reward += r
                if self.env.is_finished():
                    break
            
            trials_results.append(reward)
        
        return np.mean(trials_results), np.std(trials_results)
            

            

