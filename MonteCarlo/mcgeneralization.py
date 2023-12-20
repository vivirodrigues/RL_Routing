import osmnx as ox
import argparse
import networkx as nx
import mcagent
import approximator
import environment
from utils import *
import pandas as pd
from time import time


def execute(N0, se, env, type_env, rewards):

    n_episodes_list = [500, 1000, 2000, 5000]
    reach_list = []
    results = []

    # add um grafico q varia o min epsilon com o step-size e etc


    #for n_episodes in n_episodes_list:

    env.reset()
    # agent = approximator.MCAgent_approx(env, seed=seeds[0], gamma= 0.9, min_epsilon=0.5, n_episodes=n_episodes, max_steps = 200)
    # agent = mcagent.MCAgent(env, seed=seeds[0], gamma = 0.9, min_epsilon=0.7, max_steps=1000, n_episodes=n_episodes)

    start = time()
    agent = mcagent.MCAgent(env, seed=se, gamma = 0.9, min_epsilon=N0, max_steps=1000, n_episodes=5000, e_decay_exponentially = False, alpha = True)
    agent.train()
    end = time()

    n_reached = 0
    n_possible = len(G.nodes)

    for i in range(len(G.nodes)):
        route = agent.route_to_target(i, target)
        cost1 = agent.route_cost(env) #agent.route_to_target(G, i, target)
        # print('cost', cost)
        try:
            opt_cost = nx.shortest_path_length(G, i, target, weight="length")
            #print('opt cost', opt_cost)
            # if cost < np.inf:
            if route[-1] == target:
                n_reached += 1


        except:
            n_possible -= 1

    # reach_list.append(n_reached / n_possible)
    results.append(
        {"seed": se,
         "N0": N0,
         "cost": agent.route_cost(env),
         "time": end - start,
         "reached": n_reached/ n_possible,
         "rewards": agent.episode_rewards,
         "route": agent.route_to_target(env.source, env.target),
         "optimal": cost1/opt_cost,
         }
    )
    file_name = "data/results/monte_carlo/generalization/" + type_env + "_" + rewards + "_" + str(N0) + "_" + str(se) + ".json"
    pd.DataFrame(results).to_json(file_name)
    print("Saving", results, file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execution parameters')
    parser.add_argument('--seed', metavar='t', type=str, nargs=1, action='store', help='Pseudo Random Seed')
    parser.add_argument('--value', metavar='t', type=str, nargs=1, action='store', help='Value')
    parser.add_argument('--testing_agent_param_name', metavar='t', type=str, nargs=1, action='store', help='testing agent param name')
    parser.add_argument('--type_env', metavar='t', type=str, nargs=1, action='store', help='type environment')
    parser.add_argument('--rewards', metavar='t', type=str, nargs=1, action='store', help='type rewards')
    args = parser.parse_args()

    seed = int(args.seed[0].split('/')[-1])
    value = int(args.value[0].split('/')[-1])
    testing_agent_param_name = args.testing_agent_param_name[0].split('/')[-1]
    type_env = args.type_env[0].split('/')[-1]
    rewards = args.rewards[0].split('/')[-1]
    # print(seed, value, testing_agent_param_name, type_env, rewards)

    try:
        G = ox.load_graphml('data/grafo.graphml')
    except:
        G = ox.graph_from_address('Campinas, São Paulo', network_type='drive')
        G = nx.convert_node_labels_to_integers(G)
        ox.save_graphml(G, 'data/grafo.graphml')

    for u, v, k, data in G.edges(keys=True, data=True):
        length_value = float(data['length'])

        data['weight'] = length_value

    seeds = [960703545, 1277478588, 1936856304, 186872697, 1859168769, 1598189534, 1822174485, 1871883252, 694388766,
             188312339, 773370613, 2125204119, 2041095833, 1384311643, 1000004583, 358485174, 1695858027, 762772169,
             437720306, 939612284]

    np.random.seed(seeds[0])
    source = 507
    target = 235

    env = environment.Environment(G, source, target, rewards, type_env)

    execute(value, seed, env, type_env, rewards)
