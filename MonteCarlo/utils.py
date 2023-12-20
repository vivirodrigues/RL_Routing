import json
import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def argmax(Q, s):
    # pares estado-ação que tem estado = s
    pares = list(filter(lambda step: step[0] == s, list(Q.keys())))
    sub_Q = {k: v for k, v in Q.items() if k in pares}

    top_value = float("-inf")
    ties = []

    for i in sub_Q.items():
        # if a value in q_values is greater than the highest value update top
        if i[1] > top_value:
            top_value = i[1]

    # if a value is equal to top value add to ties
    ties = list(filter(lambda step: step[1] == top_value, list(sub_Q.items())))

    if len(ties) > 0:
        # chose the random index of the ties
        chosen = np.random.choice(range(len(ties)))
    else:
        return 0, False

    return ties[chosen][0][1], True


# def argmax(q_values, s):

#     top_value = float("-inf")
#     ties = []

#     for i in range(len(q_values[s])):
#         # if a value in q_values is greater than the highest value update top
#         if q_values[s, i] > top_value:
#             top_value = q_values[s, i]

#     # if a value is equal to top value add the index to ties
#     ties = np.where(np.array(q_values[s, :]) == top_value)[0]

#     if len(ties) > 0:
#         # return a random selection from ties.
#         return np.random.choice(ties), True
#     else:
#         return 0, False


def write_json(content, file_name):
    with open(file_name + '.json', 'w') as json_file:
        json.dump(content, json_file, separators=(',', ':'), ensure_ascii=False, sort_keys=True, indent=4)


def open_file(name_file):
    try:
        f = open('../' + name_file, "r")
        data = json.loads(f.read())
        f.close()
    except:
        data = 0
        pass

    return data


def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def average_cost(files_result, directory, confidence=0.95):
    all = []
    err = []

    for a in files_result:
        costs = []
        data = dict(open_file(directory + a))
        # print(data)
        for i in list(data.keys()):
            costs.append(float(data.get(str(i))))
        m, h = mean_confidence_interval(costs, confidence)
        all.append(m)
        err.append(h)
    return all, err


def read(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def plot_all_paths_policy(G, policy, source, target):
    """_summary_

    :param G: network
    :type G: networkx.Graph
    :param policy: dict with nodes in keys and the action in values
    :type policy: dict
    :param source: source node
    :type source: int
    :param target: target node
    :type target: int
    """

    plt.scatter(
        [G.nodes[i]["x"] for i in G.nodes],
        [G.nodes[i]["y"] for i in G.nodes],
        alpha=0.25
    )

    for i in G.nodes:
        dest = policy[i]

        # verify if there is a edge between i and dest, if not, continue
        if (i, dest) not in G.edges:
            continue

        plt.arrow(
            G.nodes[i]["x"],
            G.nodes[i]["y"],
            (G.nodes[dest]["x"] - G.nodes[i]["x"]) * 0.7,
            (G.nodes[dest]["y"] - G.nodes[i]["y"]) * 0.7,
            alpha=0.8,
            width=0.0001
        )

    # plot the source and target
    plt.scatter(
        [G.nodes[source]["x"], G.nodes[target]["x"]],
        [G.nodes[source]["y"], G.nodes[target]["y"]],
        color=["green", "red"],
        alpha=1,
        s=100
    )

    # add legend to source and target with the color

    red_patch = mpatches.Patch(color='red', label='Target')
    green_patch = mpatches.Patch(color='green', label='Source')
    plt.legend(handles=[red_patch, green_patch])

    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_graph(G, source=None, target=None):
    plt.scatter(
        [G.nodes[i]["x"] for i in G.nodes],
        [G.nodes[i]["y"] for i in G.nodes],
    )

    try:
        for u, v, m in G.edges:
            if m == 0:
                plt.plot(
                    [G.nodes[u]["x"], G.nodes[v]["x"]],
                    [G.nodes[u]["y"], G.nodes[v]["y"]],
                    color="black",
                    alpha=0.25,
                )
    except:
        for u, v in G.edges:
            plt.plot(
                [G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="black",
                alpha=0.25,
                #s = 5,
            )

    if source is not None and target is not None:
        plt.scatter(
            [G.nodes[source]["x"], G.nodes[target]["x"]],
            [G.nodes[source]["y"], G.nodes[target]["y"]],
            color=["green", "red"],
            alpha=1,
            s=100,
        )

        red_patch = mpatches.Patch(color="red", label="Target")
        green_patch = mpatches.Patch(color="green", label="Source")
        plt.legend(handles=[red_patch, green_patch])

    plt.xticks([])
    plt.yticks([])


def plot_few_steps_policy(G, policy, source, target, steps=5):
    """Plot the policy for a sequence of steps, i.e., from an initial position, plot an arrow indicating the action to take, and repeat this process for the next state.

    :param G: network
    :type G: networkx.Graph
    :param policy: dict with nodes in keys and the action in values
    :type policy: dict
    :param source: source node
    :type source: int
    :param target: target node
    :type target: int
    :param steps: number of steps to plot, defaults to 5
    :type steps: int, optional
    """

    plot_graph(G, source, target)


    state = source
    for i in range(steps):
        dest = policy[state]

        # verify if there is a edge between i and dest, if not, continue
        if (state, dest) not in G.edges:
            continue

        plt.arrow(
            G.nodes[state]["x"],
            G.nodes[state]["y"],
            (G.nodes[dest]["x"] - G.nodes[state]["x"]) * 0.95,
            (G.nodes[dest]["y"] - G.nodes[state]["y"]) * 0.95,
            alpha=0.8,
            width=0.0001,
            #s=5,
        )
        state = dest
    #plt.show()


def plot_value_func(G, source, target, value_func):
    plot_graph(G, source, target)
    plt.scatter(
        [G.nodes[i]["x"] for i in G.nodes],
        [G.nodes[i]["y"] for i in G.nodes],
        c=value_func,
        cmap="YlOrRd",
    )


def create_df_gamma(reward_method, env_method, gamma, seeds):
    data = []
    for i in gamma:
        for j in seeds:
            file = "data/results/monte_carlo/" + env_method + "/" + reward_method + "/gamma_" + str(i) + "_" + str(
                j) + ".json"
            info = read(file)
            info = {i: tuple(k.items())[0][1] for i, k in info.items()}
            data.append(info)
    return pd.DataFrame.from_dict(data)


def create_df_rl(reward_method, env_method, epsilon, seeds):
    data = []
    for i in epsilon:
        for j in seeds:
            file = "data/results/monte_carlo/" + env_method + "/" + reward_method + "/min_epsilon_" + str(
                i) + "_" + str(j) + ".json"
            info = read(file)
            info = {i: tuple(k.items())[0][1] for i, k in info.items()}
            data.append(info)
    return pd.DataFrame.from_dict(data)


def analysis_results(reward_method, env_method, gamma, epsilon, seeds):
    results_lr = create_df_rl(reward_method, env_method, epsilon, seeds)
    results_gamma = create_df_gamma(reward_method, env_method, gamma, seeds)
    results_lr["rewards"] = results_lr.rewards.apply(np.mean)
    results_gamma["rewards"] = results_gamma.rewards.apply(np.mean)

    results_lr = results_lr.groupby("min_epsilon").agg({
        "cost": ["mean", "std"],
        "time": ["mean", "std"],
        "rewards": ["mean", "std"]
    })
    results_lr.columns = ["_".join(col) for col in results_lr.columns]
    results_lr = results_lr.reset_index()

    results_gamma = results_gamma.groupby("gamma").agg({
        "cost": ["mean", "std"],
        "time": ["mean", "std"],
        "rewards": ["mean", "std"]
    })
    results_gamma.columns = ["_".join(col) for col in results_gamma.columns]
    results_gamma = results_gamma.reset_index()
    plot_results_experiments(results_lr, results_gamma, f"{reward_method.title()} reward and {env_method.title()} environment")


def plot_results_experiments(results_lr, results_gamma, title):
    """Plot the results of the experiments varying learning rate and gamma.

    Parameters
    ----------
    results_lr : pd.DataFrame
        Dataframe with columns learning_rate, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    results_gamma : pd.DataFrame
        Dataframe with columns gamma, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    title : str
        Title of the plot
    """

    def plot_line_plot_err(df, param, cost, ax):
        ax.plot(df[param], df[f"{cost}_mean"], color="black")
        ax.fill_between(
            df[param],
            df[f"{cost}_mean"] - df[f"{cost}_std"],
            df[f"{cost}_mean"] + df[f"{cost}_std"],
            color="black",
            alpha=0.2
        )
        ax.set_xlabel(param)
        ax.set_ylabel(cost)
        ax.set_title(f"{cost} by {param}")
        ax.grid()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    axs = axs.flatten()

    plot_line_plot_err(results_lr, "min_epsilon", "cost", axs[0])
    plot_line_plot_err(results_lr, "min_epsilon", "time", axs[1])
    plot_line_plot_err(results_lr, "min_epsilon", "rewards", axs[2])

    plot_line_plot_err(results_gamma, "gamma", "cost", axs[3])
    plot_line_plot_err(results_gamma, "gamma", "time", axs[4])
    plot_line_plot_err(results_gamma, "gamma", "rewards", axs[5])

    plt.suptitle(title, fontsize=16)
    plt.show()


def plot_results_experiments3(results_lr, title):
    """Plot the results of the experiments varying learning rate and gamma.

    Parameters
    ----------
    results_lr : pd.DataFrame
        Dataframe with columns learning_rate, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    results_gamma : pd.DataFrame
        Dataframe with columns gamma, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    title : str
        Title of the plot
    """
    better_names = {
        "cost": "Path cost",
        "time": "Computing time",
        "rewards": "Rewards",
        "N0": "N0",
        "gamma": "Discount factor (γ)",
    }

    def plot_line_plot_err(df, param, cost, ax):
        ax.plot(df[param], df[f"{cost}_median"], color="black")
        ax.fill_between(
            df[param],
            df[f"{cost}_1q"],
            df[f"{cost}_3q"],
            color="black",
            alpha=0.2
        )
        ax.set_xlabel(better_names[param])
        ax.set_ylabel(better_names[cost])
        ax.set_title(f"{better_names[param]} x {better_names[cost]}")
        ax.grid()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), sharey="col", sharex="row")
    axs = axs.flatten()

    plot_line_plot_err(results_lr, "N0", "cost", axs[0])
    plot_line_plot_err(results_lr, "N0", "time", axs[1])
    plot_line_plot_err(results_lr, "N0", "rewards", axs[2])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_results_experiments4(results_lr, title):
    """Plot the results of the experiments varying learning rate and gamma.

    Parameters
    ----------
    results_lr : pd.DataFrame
        Dataframe with columns learning_rate, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    results_gamma : pd.DataFrame
        Dataframe with columns gamma, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    title : str
        Title of the plot
    """
    better_names = {
        "cost": "Path cost",
        "time": "Computing time",
        "rewards": "Rewards",
        "N0": "N0",
        "reached": "Nodes that finds the target [%]",
        "optimal": "Route cost / optimal cost",

    }

    def plot_line_plot_err(df, param, cost, ax):
        ax.plot(df[param], df[f"{cost}_median"], color="black")
        print(df[param], df[f"{cost}_median"], param, cost)
        ax.fill_between(
            df[param],
            df[f"{cost}_1q"],
            df[f"{cost}_3q"],
            color="black",
            alpha=0.2
        )
        ax.set_xlabel(better_names[param])
        ax.set_ylabel(better_names[cost])
        ax.set_title(f"{better_names[param]} x {better_names[cost]}")
        ax.grid()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey="col", sharex="row")
    axs = axs.flatten()

    plot_line_plot_err(results_lr, "N0", "reached", axs[0])
    plot_line_plot_err(results_lr, "N0", "optimal", axs[1])
    # plot_line_plot_err(results_lr, "N0", "rewards", axs[2])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def analysis_results3(reward_method, env_method, N0, seeds):

    data = []
    for i in N0:
        for j in seeds:
            file = "data/results/monte_carlo/generalization/"+ env_method +"_" + reward_method +"_" + str(i) +"_" + str(j) + ".json"
            info = read(file)
            info = {i: tuple(k.items())[0][1] for i, k in info.items()}
            data.append(info)
    results_lr = pd.DataFrame.from_dict(data)
    #results_lr = pd.read_json(f"data/results/monte_carlo/generalization/{env_method}_{reward_method}_{N0}_{se}.json")
    results_lr["rewards"] = results_lr.rewards.apply(np.mean)
    print(results_lr.columns)
    results_lr_agg = []

    lr_values = results_lr.N0.unique()
    for lr in lr_values:
        results_lr_filter = results_lr[results_lr.N0 == lr]
        # calculate median, 1q and 3q from each column'
        results_lr_agg.append({
            "N0" : lr,
            "cost_median" : np.nanmedian(results_lr_filter.cost),
            "cost_1q" : np.nanquantile(results_lr_filter.cost, 0.25),
            "cost_3q" : np.nanquantile(results_lr_filter.cost, 0.75),
            "time_median" : np.nanmedian(results_lr_filter.time),
            "time_1q" : np.nanquantile(results_lr_filter.time, 0.25),
            "time_3q" : np.nanquantile(results_lr_filter.time, 0.75),
            "rewards_median" : np.nanmedian(results_lr_filter.rewards),
            "rewards_1q" : np.nanquantile(results_lr_filter.rewards, 0.25),
            "rewards_3q" : np.nanquantile(results_lr_filter.rewards, 0.75),
            "reached_median": np.nanmedian(results_lr_filter.reached),
            "reached_1q": np.nanquantile(results_lr_filter.reached, 0.25),
            "reached_3q": np.nanquantile(results_lr_filter.reached, 0.75),
            "optimal_median": np.nanmedian(results_lr_filter.optimal),
            "optimal_1q": np.nanquantile(results_lr_filter.optimal, 0.25),
            "optimal_3q": np.nanquantile(results_lr_filter.optimal, 0.75),
        })

    results_lr_agg = pd.DataFrame(results_lr_agg)

    better_names = {
        "unit" : "Unit",
        "weighted" : "Weighted",
        "deterministic" : "Deterministic",
        "stochastic" : "Stochastic",
        "distance" : "Distance"
    }

    print(results_lr_agg)
    plot_results_experiments3(results_lr_agg, f"Parameter evaluation\n{better_names[reward_method]} reward and {better_names[env_method]} environment")
    plot_results_experiments4(results_lr_agg,
                              f"Parameter evaluation\n{better_names[reward_method]} reward and {better_names[env_method]} environment")
