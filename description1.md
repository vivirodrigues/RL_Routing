# RL_Routing

This work presents optimized routes between two nodes in a geographic network, comparing the performance of different reinforcement learning methods.
The shortest path between two nodes in a network may suggest the route with the minimum distance, reduced fuel consumption, or even a shorter time requirement. For example, we need to identify a path from the source to the target node that covers fewer meters to minimize the distance.
The algorithms implemented where: Monte Carlo, QLearning, SARSA and DQN.

## Using

This implementation was developed using Conda and Colab. The depencies for this project are numpy, matplotlib, osmnx and torch. To install it, run:

`pip install numpy matplotlib osmnx torch`

All implementations are inside notebooks that are self-contained (they do not use external scripts) and organized in folders with the name according to the method. 

`description.md` presents an complete discussion of the project.
