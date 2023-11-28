# Reinforcement Learning for Routing

Our objective is to use reinforcement learning to learn a policy of routes for a trash collector truck. An optimized path can help in obtaining a faster and more efficient trash collection. The problem can be modeled as finding the shortest path between two nodes in a graph, which have already been previously solved using other optimization approachs, heuristics and graph algorithms. However we want to apply reinforcement learning in this problem both for educational purposes and compare it with other approaches The problem can be defined as follow:

**Definition.**: With a weighted directed graph $G = (V, E)$, and a starting node $v$, find the shortest path to node $u$, i.e., the subset of $E$ that connections $v$ to $u$ and has minimal sum of edges weights.

To solve this problem with reiforcement learning, we need to identify the actions, states, etc.

**Reinforcement learning formulation**:
- Episodic, finishes when arrive on destination node or in a node without leaving edges.
- The discretization occurs at the vertex of the graph, i.e., the crossings of the streets in which the driver needs to make a decision.
- States: $n$ states $v_i \in V$ , each is a node of the graph.
- Actions: $n$ action, each one correspond to moving to a specific node. It is important to note that the possible actions are dependent by each step. The action of moving to node $u$, if the state is node $v$, can only be made if there is an edge from $v$ to $u$.
- There are two considered reward schemes:
    - Unit: receives reward 10000 if reach goal, reward -1 for chosing to stay at the same position, -1000 if leaves to an dead end, and 0 for other possible actions (valid movements in the graph).
    - Weighted: receives reward 10000 if reach goal, reward -1 for chosing to stay at the same position, -1000 if leaves to an dead end, and $-w_{u, v}$ for other possible actions (valid movements in the graph). $w_{u, v}$ is the weight of the edge between $u$ and $v$ normalized by the maximum value of edges.


**Enviroment:** Our final goal is to be able to incorporate our learned algorithm to [SUMO](https://eclipse.dev/sumo/), a realistic simulation of urban mobility. However, to facilitate the initial study, we opted to implement an enviroment ourselves using only numpy functions and networkx to deal with the graph functions. This enviroment has the capabilities of performing deterministic or stochastic steps. It verifies if the action is feasible for the current state of the simulation, computes the following state from the action, and returns the reward.


## Q-learning

Q-learning was implemented using the epsilon-greedy policy, with linear decay, achieving a value $\varepsilon_{\textrm{min}}$ at the final iteration. The learning rate $\alpha$ was a parameter, and was kept fixed during training (without learning rate decay). The Q-matrix was initialized with all values equal to $0$ and $-\infty$ in pairs $(s, a)$ that are not valid actions, i.e., pairs that there isn't a edge leaving node $s$ to $a$. The value $-\infty$ was used so that these pairs are not selected as the argmax values. 


### Experiments
Fixed source, target, run with 20 seeds
- Deterministic
    - Learning rate (two reward methods)
    - Gamma (two reward methods)
    - Min epsilon
- Stochastic
    - Learning rate (two reward methods)
    - Gamma (two reward methods)
    - Min epsilon

$\gamma = \{ 0.1, 0.25, 0.5, 0.9, 0.99, 1\}$


Vary source, target with 20 seeds
Fixed parameters





