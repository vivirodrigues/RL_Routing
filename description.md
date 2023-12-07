# Reinforcement Learning for Routing

Our objective is to use reinforcement learning in the problem of path fiding for a trash collector truck. An optimized path can help in obtaining a faster and more efficient trash collection. The problem can be modeled as finding the shortest path between two nodes in a graph, which have already been previously solved using other optimization approachs, heuristics and graph algorithms. However we want to apply reinforcement learning in this problem both for educational purposes and compare it with other approaches.

 The problem can be defined as follow:

**Definition.**: With a weighted directed graph $G = (V, E)$, and a starting node $v$, find the shortest path to node $u$, i.e., the subset of $E$ that connections $v$ to $u$ and has minimal sum of edges weights.

## Reinforcement learning formulation

### Markov Decision Process

- Episodic, episode ends when the agent arrives on destination node or in a node without leaving edges.
- The time discretization occurs at the nodes of the graph, i.e., the crossings of the streets in which the agent needs to make a decision of which direction to follow.
- States: $n$ states $v_i \in V$ , each is a node of the graph.
- Actions: $n$ actions, each one correspond to moving to a specific node. It is important to note that the possible actions are dependent by each state. The action of moving to node $u$, if the state is node $v$, can only be made if there is an edge from $v$ to $u$.

### Rewards

There are two considered reward schemes:

- *Unit*: receives reward 10000 if reach goal, reward -1 for chosing to stay at the same position, -1000 if leaves to an dead end, and 0 for other possible actions (valid movements in the graph).
- *Weighted*: receives reward 10000 if reach goal, reward -1 for chosing to stay at the same position, -1000 if leaves to an dead end, and $-w_{u, v}$ for other possible actions (valid movements in the graph). $w_{u, v}$ is the weight of the edge between $u$ and $v$ normalized by the maximum weight of edges.

These reward setups were designed with the intent to [induzir] some behaviors in the agent. The negative reward $-1$ is so that the agent does not learn to stay at the same position. The negative reward $-1000$ at dead ends is also to enforce the agent to avoid dead ends with few visits. 

In the *unit* reward scheme there is not use of the edges weights, so there is no information for the agent to learn the optimal path, the only objective is to reach the goal.  In the *weighted* reward scheme, we incorporate the edge weights as negatives rewards. As the objective is to obtain the path with shortest path, we need to minimize the sum of the edges weights, or maximized the negative of the sum. This could teach the model to stay at the same position, as both staying at the same position or selecting a new node will have negative weights. However, the weights are normalized, and only the maximum $w_{u, v}$ will be equal $-w_{u, v} = -11$ (the reward of staying in the same node).


### Enviroment

Our final goal is to be able to incorporate our learned algorithm to [SUMO](https://eclipse.dev/sumo/), a realistic simulation of urban mobility. However, to facilitate the initial study, we opted to implement our own enviroment using only numpy and networkx to support graph objects. 

The enviroment keeps the current state of the simulation and has the `step` method. This method receives an action and according to the current state, returns the new state of performing this action. It also returns the reward obtained from the action. If the action is possible, i.e., there is a edge from $v$ to $u$, it perform the step and update the state to $u$, otherwise the state stays at $v$. 

The enviroment can be deterministic or stochastic. The stochastic step has two extra details. First, an action has a random probability of not being possible (we used 5%). Secondly, we add a random gaussian noise to the weights of the edges, this noise is sampled at each step, so the same edge will have different weights at different iterations. Our intention with this stochastic implementation is to simulate the uncertainty of transit, some streets can be randomly not accessible, and the cost of going trought a street can also have different values dependending on the day, time of the day, climate, etc.

## Monte-Carlo

## Q-learning

### Implementation details

Q-learning was implemented using the epsilon-greedy policy, with linear decay, achieving a value $\varepsilon_{\textrm{min}}$ at the final iteration. Both the $\varepsilon_{\textrm{max}}$ and $\varepsilon_{\textrm{min}}$ are parameters of the agent, however, there is no reason to have $\varepsilon_{\textrm{max}} \neq 1$. Other parameter was the learning rate $\alpha$, and was kept fixed during training (without learning rate decay). Good values for the learning rate are in the interval $[0, 1]$. $\gamma$, maximum number of steps per episode and number of episodes were the remaining parameters.

The Q-matrix was initialized with all values equal to $0$ and $-\infty$ in pairs $(s, a)$ that are not valid state-actions, i.e., pairs that there isn't a edge leaving node $s$ to $a$. The value $-\infty$ was used so that these pairs are not selected as the argmax values in the greedy policy. 


### Experiments

#### Parameters analysis

Our agent has to main parameters that need to be considered, the learning rate $\alpha$ and the weight of future rewards $\gamma$. To evaluate the adequate values for these parameters, we perform an experiment to evaluate how the parameter values will impact the path cost, computational time and mean reward. It is also important to analyse how the agent work in the four possible scenarios of enviroment and reward: (deterministic, unit), (deterministic, weighted), (stochastic, unit), and (stochastic, weighted).

The $\alpha$ values tested are $\{0.05, 0.1, 0.3, 0.5, 0.7\}$ and the $\gamma$ values are $\{0.1, 0.25, 0.5, 0.9, 0.99\}$. When varying $\alpha$, $\gamma = 0.99$, when varying $\gamma$, $\alpha = 0.7$. The experiments were performed with $1000$ episodes of $1000$ steps at max each. For each of the $\alpha$ and $\gamma$ values, we training was executed 20 times using the same pair of source and target, however using a different random seed. We saved the cost of the path found by the policy, the training time and the average reward for episode.


#### Generalization

After identying a good parameter values, it is important to evaluate if our agent is able to generalize to different pairs of source and target. This was performed with two analysis, first, we selected 20 random source and target pairs, and using the optimal parameters found, trained the agent to find the path for each of the source, target pair. We saved the duration of training, and average reward per episode and the ratio between the cost of the path found by our agent and the cost of the optimal path.


The next analysis is to evaluate if an agent trained in a source and target pair can find a path from any of the states to the target, and also if the path found is optimal. To do that, with an trained agent, we selected all possible states and calculated the route find by the policy to the target. We saved the information if the route was found and also the ratio of the cost found and the optimal cost.





## SARSA

## DQN


## Contributions

- Enviroment: Giovani and Vitoria
- Monte Carlo: Vitoria
- QLearning: Giovani
- SARSA: Marcos
- DQN: 