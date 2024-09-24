import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .graph_utils import BA_model, normalize_out_degree
from .authors_prep import plot_network

def plot_network_colored(G, nodes):
    graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
    recovered_nodes = np.where(nodes == "s")[0]
    infected_nodes = np.where(nodes == "i")[0]
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx_nodes(graph, pos, nodelist=recovered_nodes, node_color='#A0CBE2')
    nx.draw_networkx_nodes(graph, pos, nodelist=infected_nodes, node_color='red')
    nx.draw_networkx_edges(graph, pos, alpha=0.4)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='verdana')
    plt.show()


def initial_infection(allnodes, initial_infec = None):
    """
    Initializes infection status of a population. Each individual can either be infected "i"
    or susceptible "s". By default, one individual is initially randomly infected unless a number
    is provided.
    :param allnodes: Integer representing the total number of individuals in the population.
    :param initial_infec: Optional, number of individuals initially infected. Default is 1.
    :return: An array initial_status with the initial infection status
    """
    # init all individuals as susceptible
    initial_status = np.full(allnodes, 's')
    # determine the number of individuals to infect
    if initial_infec is None:
        initial_infec = 1
    # randomly select individuals to be infected
    initially_infected_nodes = np.random.choice(allnodes, size = initial_infec, replace = False)
    #update status
    initial_status[initially_infected_nodes] = "i"

    return initial_status




def SIS_step(G, nodes_str, mu, beta):
    """
    Performs an SIS model step to update infection probabilities for each individual.
    Probability of staying infected or becoming infected depends on the individual's current state,
    the states of their neighbors, the network's structure, the recovery rate and the infection rate.
    :param G: A square numpy array where non-zero values indicate connections between individuals.
    :param nodes_str: An array of characters ("i" for infected, "s" for susceptible)
    :param mu: Probability of an infected individual recovering.
    :param beta: Probability of infection spreading from an infected individual to a susceptible one.
    :return: A numpy array with updated infection probabilities for each individual.
    """
    # initialize probabilities array with zeros for each individual
    new_prob = np.zeros(len(nodes_str))
    # convert status, string array into a binary array
    infection_status = np.where(nodes_str == 'i', 1, 0)

    # iterate through each node in network to calculate new chance of infection
    for node_index in range(len(G)):
        # start with base infection chance
        infection_chance = 1
        # iterate through each neighbor of each node to adjust infection rate based on link
        for neighbor in range(len(G)):
            if G[neighbor, node_index] > 0: #if connected
                # adjust infection chance based on connection and neighbor's infected status
                infection_chance *= (1 - beta * G[neighbor, node_index] * infection_status[neighbor])
        new_prob[node_index] = (1 - infection_chance) * (1 - infection_status[node_index]) + \
                               (1 - mu) * infection_status[node_index] + \
                               mu * (1 - infection_chance) * infection_status[node_index]

    return new_prob




def new_state_nodes( prob, nodes ):
    """
    Updates infection status of individuals in a population based on new probabilites.
    Probabilites are computed using an SIS model step.
    :param prob: Array of probabilities indicating chance of each individual being infected.
    :param nodes: Array representing the current state of each individual.
    :return: Array representing the updated state of each individual after the step.
    """
    # update each individual's state based on a random threshold
    for i, p in enumerate(prob):
        # generate random threshold for infection
        threshold = np.random.rand()
        # update individual's state
        nodes[i] = "i" if p > threshold else "s"
    return nodes


def p_vs_steps(rmse_threshold = False):
    networks_to_generate = 10
    num_of_simulations = 50
    num_nodes = 40
    init_infected_sizes = [2, 8, 40]
    R0_values = [(1, 0.5), (0.1, 0.8)] # (mu, beta) pairs

    results = np.zeros((len(init_infected_sizes), len(R0_values), num_of_simulations))
    infection_ratio_history = []

    for i, init_infected in enumerate(init_infected_sizes):
        for j, (mu, beta) in enumerate(R0_values):
            for network_index in range(networks_to_generate):
                network = BA_model(3, 2, num_nodes)
                network = normalize_out_degree(network)
                nodes = initial_infection(num_nodes, init_infected)
                infection_ratio_history.clear()

                for sim_step in range(num_of_simulations):
                    if sim_step > 0:
                        prob = SIS_step(network, nodes, mu, beta)
                        nodes = new_state_nodes(prob, nodes)
                        infection_count = np.sum(nodes == "i")/num_nodes
                        results[i,j,sim_step] += infection_count
                        infection_ratio_history.append(infection_count)

                        if rmse_threshold and len(infection_ratio_history) >= 6:
                            if rmse(infection_ratio_history[-3:], infection_ratio_history[-6:-3]) == 0:
                                break


            results[i,j,:] /= networks_to_generate



    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red"]
    labels = ["R0 < 1", "R0 > 1"]
    linestyles = ["--", "-"]

    for i, infected_num in enumerate(init_infected_sizes):
        for j, (beta, mu) in enumerate(R0_values):
            linestyle = linestyles[j]
            plt.plot(results[i, j, :], color=colors[i], linestyle=linestyle,
                     label=f"Initial Infected: {infected_num}, {labels[j]}")

    plt.xlabel("Steps")
    plt.ylabel("p (Ratio of Infected Individuals)")
    plt.title("Evolution of Infection Ratio Over Time")
    plt.legend()
    plt.show()


def p_vs_beta(rmse_threshold = False):
    num_networks = 10
    num_simulations = 50
    num_nodes = 40
    beta_values = np.linspace(0.01, 2, 25)
    mu_values = [0.1, 0.2, 0.3, 0.4]
    results = np.zeros((len(mu_values), len(beta_values)))

    for i, mu in enumerate(mu_values):
        for j, beta in enumerate(beta_values):
            infection_counts = []

            for network_index in range(num_networks):
                network = BA_model(3, 2, num_nodes)
                network = normalize_out_degree(network)
                current_nodes = initial_infection(num_nodes)

                sim_infection_counts = []

                for simulation in range(num_simulations):
                    prob = SIS_step(network, current_nodes, mu, beta)
                    new_nodes = new_state_nodes(prob, current_nodes)
                    infection_count_current = np.sum(new_nodes == "i") / num_nodes
                    sim_infection_counts.append(infection_count_current)

                    if rmse_threshold and len(sim_infection_counts) >= 6:
                        if rmse(sim_infection_counts[-3:], sim_infection_counts[-6:-3]) == 0:
                            break

                infection_counts.extend(sim_infection_counts)

            results[i, j] = np.mean(infection_counts)

    # Plotting results
    plt.figure(figsize=(10, 7))
    for i, mu in enumerate(mu_values):
        plt.plot(beta_values, results[i, :], label=f'Mu={mu}')
    plt.xlabel('Beta')
    plt.ylabel('Average fraction of infected individuals (ρ)')
    plt.title('ρ vs Beta')
    plt.legend()
    plt.show()


def p_vs_r0(rmse_threshold = False):
    num_networks = 10
    num_simulations = 50
    betas = np.linspace(0.01, 2, 25)
    mu = 0.3
    num_nodes = 40
    starting_nodes = range(2, 11, 2)

    for m0 in starting_nodes:
        m = m0 - 1
        results = np.zeros(len(betas))

        for network_index in range(num_networks):
            network = BA_model(m0, m, num_nodes)
            network = normalize_out_degree(network)

            previous_infection_counts = None
            for i, beta in enumerate(betas):
                r0 = beta / mu
                infection_counts = []

                nodes = initial_infection(num_nodes)
                for simulation in range(num_simulations):
                    prob = SIS_step(network, nodes, mu, beta)
                    nodes = new_state_nodes(prob, nodes)
                    infection_count = np.sum(nodes == "i") / num_nodes
                    infection_counts.append(infection_count)

                    if rmse_threshold and len(infection_counts) >= 6:
                        if rmse(infection_counts[-3:], infection_counts[-6:-3]) == 0:
                            break

                avg_infection_count = np.mean(infection_counts)
                results[i] += avg_infection_count

        results /= num_networks
        plt.plot(betas / mu, results, label=f"m0 = {m0}, m = {m}")

    plt.xlabel("R0 (Beta / Mu)")
    plt.ylabel("Average fraction of infected individuals (p)")
    plt.title("p vs. R0 for different network topologies")
    plt.legend()
    plt.grid(True)
    plt.show()




def rmse(current, previous):
    """
    Calculate the Root Mean Square Error (RMSE) between two sets of observations.
    In an attempt to overcome the incompatibilites between this function and our graph plotting
    functions logic, we decided to calculate the RMSE, in each of our graphs, in each valid simulation,
    between the most recent 3 steps and the 3 steps before. We also set the threshold at 0. Anything else
    was not compatible with our code. None of our graphs converge within the range of simulation regardless.
    :param current: Numpy array of current observations.
    :param previous: Numpy array of previous observations.
    :return: The calculated RMSE
    """
    current, previous = np.array(current), np.array(previous)
    mse = np.sqrt(np.mean((current-previous)**2))
    return mse

#when the mse is below a certain threshold, the iterations cant stop as convergence would have been achieved

