import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_all_paths_policy(G, policy, source, target):
    """Plot the policy for all states, i.e., for each state, plot an arrow indicating the action to take.

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
        alpha=0.25,
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
            width=0.0001,
        )

    # plot the source and target
    plt.scatter(
        [G.nodes[source]["x"], G.nodes[target]["x"]],
        [G.nodes[source]["y"], G.nodes[target]["y"]],
        color=["green", "red"],
        alpha=1,
        s=100,
    )

    # add legend to source and target with the color

    red_patch = mpatches.Patch(color="red", label="Target")
    green_patch = mpatches.Patch(color="green", label="Source")
    plt.legend(handles=[red_patch, green_patch])

    plt.xticks([])
    plt.yticks([])
    plt.show()


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

    plt.scatter(
        [G.nodes[i]["x"] for i in G.nodes],
        [G.nodes[i]["y"] for i in G.nodes],
        alpha=0.25,
    )

    # plot the edges
    for u, v, m in G.edges:
        if m == 0:
            plt.plot(
                [G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="black",
                alpha=0.25,
            )

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
        )
        state = dest

    # plot the source and target
    plt.scatter(
        [G.nodes[source]["x"], G.nodes[target]["x"]],
        [G.nodes[source]["y"], G.nodes[target]["y"]],
        color=["green", "red"],
        alpha=1,
        s=100,
    )

    # add legend to source and target with the color

    red_patch = mpatches.Patch(color="red", label="Target")
    green_patch = mpatches.Patch(color="green", label="Source")
    plt.legend(handles=[red_patch, green_patch])

    plt.xticks([])
    plt.yticks([])
