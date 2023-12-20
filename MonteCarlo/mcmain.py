import osmnx as ox
import argparse
import networkx as nx
import mcagent
import approximator
import environment
from utils import *
import pandas as pd
from time import time
import matplotlib.pyplot as plt


def experiment(se, env, value, testing_agent_param_name, fixed_agent_params, type_env='deterministic', rewards='weighted', linear_approximator = False):
    if linear_approximator:
        fixed_agent_params['gamma'] = 0.9
        fixed_agent_params['min_epsilon'] = 0
        # fixed_agent_params['e_decay_exponentially'] = False
    elif testing_agent_param_name != 'default':
        fixed_agent_params[testing_agent_param_name] = value
    else:
        fixed_agent_params['e_decay_exponentially'] = False
        fixed_agent_params['alpha'] = True
        fixed_agent_params['min_epsilon'] = 300 # 0.7

    results = []

    start = time()

    if linear_approximator:
        agent = approximator.MCAgent_approx(env, seed=se, **fixed_agent_params, )
    else:
        agent = mcagent.MCAgent(env, seed=se, **fixed_agent_params, )
    agent.train()

    end = time()

    results.append(
        {"seed": se,
         testing_agent_param_name: value,
         "cost": agent.route_cost(env),
         "time": end - start,
         "rewards": agent.episode_rewards,
         "route": agent.route_to_target(env.source, env.target),
         }
    )
    if linear_approximator:
        file_name = "data/results/monte_carlo/" + type_env + "/" + rewards + "/" + testing_agent_param_name + "_" + str(
            value) + "_" + str(se) + "_aplinear2.json"
    else:
        file_name = "data/results/monte_carlo/"+type_env+"/"+rewards+"/"+testing_agent_param_name+"_" + str(value) + "_" + str(se) + ".json"
    pd.DataFrame(results).to_json(file_name)
    print("Saving", file_name)

    # env.reset()
    # plt.plot(agent.episode_rewards)
    # plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execution parameters')
    parser.add_argument('--seed', metavar='t', type=str, nargs=1, action='store', help='Pseudo Random Seed')
    parser.add_argument('--value', metavar='t', type=str, nargs=1, action='store', help='Value')
    parser.add_argument('--testing_agent_param_name', metavar='t', type=str, nargs=1, action='store', help='testing agent param name')
    parser.add_argument('--type_env', metavar='t', type=str, nargs=1, action='store', help='type environment')
    parser.add_argument('--rewards', metavar='t', type=str, nargs=1, action='store', help='type rewards')
    args = parser.parse_args()
    
    seed = int(args.seed[0].split('/')[-1])
    value = float(args.value[0].split('/')[-1])
    testing_agent_param_name = args.testing_agent_param_name[0].split('/')[-1]
    type_env = args.type_env[0].split('/')[-1]
    rewards = args.rewards[0].split('/')[-1]

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

    print(f"The source is {source} and the target is {target}")

    env = environment.Environment(G, source, target, rewards, type_env)

    fixed_agent_params = {
        "n_episodes": 1000,
        "max_steps": 200,
    }

    experiment(seed, env, value, testing_agent_param_name, fixed_agent_params, type_env, rewards, True)
