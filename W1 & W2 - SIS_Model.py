#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:40:04 2024

@authors: Nicolau San Millán, Sander Rannik, Yago Granada, Riccardo Orsini and Rodrigo Sastré
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from colorama import Fore
from itertools import product


def print_matrix(G):
    for i, j in product(range(G.shape[0]), range(G.shape[1])):
        ending = "\n" if j == G.shape[0] - 1 else ""
        if G[i, j] == 0:
            print(Fore.WHITE, f'{round(G[i, j], 2)}', end=ending)
        if G[i, j] > 0:
            print(Fore.GREEN, f'{round(G[i, j], 2)}', end=ending)


# Draw the network
def plot_network(G):
    graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
    pos = nx.spring_layout(graph)  # Position nodes using the spring layout algorithm
    #    pos = nx.fruchterman_reingold_layout( graph )  # Position nodes using the Fruchterm R. layout algorithm
    #    pos = nx.spectral_layout( graph )              # Position nodes using specral layout algorithm
    #    pos = nx.circular_layout( graph )              # Position nodes using circular layout algorithm
    nx.draw(graph, pos, with_labels=True, node_size=100, font_size=5)
    plt.show()


# ---------------- Start Functions Part I ------------
def full_mesh(num_nodes):
    """
    Generates a complete network (fully connected graph) adjacency matrix, where each node is connected
    to every other node, except itself.

    Args:
    - num_vertices: The number of nodes in the graph.

    Returns:
    - A numpy array representing the adjacency matrix of the complete network.
    """
    # Initialize an all-ones matrix and subtract the identity matrix to remove self-loops
    network = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    return network

def compute_p(network):
    """
    Calculates the attachment probabilities for each node in a network based on the Barabási–Albert (BA) model.
    The probability is proportional to the node degree, facilitating preferential attachment where nodes with higher degrees
    are more likely to receive new links.

    Args:
    - adjacency_matrix: A numpy array representing the network's adjacency matrix.

    Returns:
    - A numpy array containing the probability of attachment for each node in the network.
    """
    # Convert the numpy adjacency matrix to a NetworkX graph for easy degree calculation
    graph = nx.from_numpy_array(network)
    # Calculate the degree of each node within the graph
    degrees = np.array([graph.degree[node] for node in graph.nodes()])
    # Normalize the degrees to calculate probabilities, ensuring the sum of all probabilities equals 1
    probabilities = degrees / degrees.sum()
    return probabilities


def nodes_selection(network, links_per_step):
    """
    Selects a specified number of unique nodes from network based on preferential
    attachment probabilities.
    :param network: Numpy array representing adjacency matrix of network.
    :param links_per_step: The number of unique nodes to select, per step.
    :return: A list of selected node indices.
    """
    # calculate selection probabilities for each node
    p = compute_p(network)
    selected_nodes = set()
    # continue selecting nodes until the desired number of non-repeated nodes is reached
    while len(selected_nodes) < links_per_step:
        # select a node based on computed probabilities
        potential_node = np.random.choice(range(network.shape[0]), p=p)
        # add selected node to a set to ensure uniqueness
        selected_nodes.add(potential_node)
    return list(selected_nodes)

def BA_model(initial_size, links, final_size):
    """
    Generates a network based on the Barabási-Albert (BA) model, starting with a fully connected mesh
    and iteratively adding nodes. Each new node connects to existing nodes based on preferential
    attachment, simulating the growth of real-world networks.

    Args:
    - initial_size: The size of the initial fully connected network.
    - links: The number of links each new node should create with existing nodes.
    - final_size: The final size of the network.

    Returns:
    - A numpy array representing the adjacency matrix of the generated network.
    """
    # Create an initial fully connected mesh
    network = full_mesh(initial_size)
    # Iteratively add new nodes to the network until the final size is reached
    for new_node in range(final_size - initial_size):
        new_node_index = network.shape[0]  # Index for the new node
        # Create a new matrix to accommodate the new node
        new_network = np.zeros((new_node_index + 1, new_node_index + 1))
        # Copy the existing network into the new matrix
        new_network[:new_node_index, :new_node_index] = network
        network = new_network
        # Select nodes for attachment based on preferential attachment
        attachment_nodes = nodes_selection(network, links)
        # Form links between the new node and selected existing nodes
        for node in attachment_nodes:
            network[new_node_index, node] = 1
            network[node, new_node_index] = 1
    return network


def normalize_out_degree(network):
    """
    Normalizes out-degrees in a network represented by an adjacency matrix.
    Ensure that, for each node, the sum of the weights of its outgoing edges is 1.
    :param network: Numpy array representing an adjacency matrix of the network.
    :return: Numpy array representing the normalized adjacency matrix.
    """
    # calculate sum of each column
    degrees = np.sum(network, axis=0)

    # We transposed matrix to normalize by columns (out-degree)
    # A new matrix is created where each column is divided by its corresponding out-degree sum.
    # avoid division by zero using where condition
    normalized_p = np.divide(network.T, degrees, where=degrees != 0)
    # convert NaN values to 0 in case of division by zero (nodes with no outgoing edges)
    network = np.nan_to_num(normalized_p).T #transpose back to original orientation
    return network


def plot_degree_histogram(network):
    graph = nx.from_numpy_array(network)
    degrees = [degree for node, degree in graph.degree()]
    plt.hist(degrees, bins=np.arange(0.5, max(degrees) + 1.5, 1), alpha=0.75, color="blue")
    # naming the x axis
    plt.xlabel('Node degree')
    # naming the y axis
    plt.ylabel('Number of nodes')
    # giving a title to the graph
    plt.title('Degree nodes plot')
    plt.grid(axis="y", alpha=0.75)

    # Draw the degree distribution plot
    # It has to be a plot with node degree as x-axis
    # and the number of nodes having a degree of x as y-axis

    plt.show()


# ---------------- Start Functions Part II ------------

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


# ---------------- End Functions Part 2 ------------

# Main
# Variable init
allnodes = 50  # final number of nodes
initialnodes = 2  # initial number of nodes
m = 2  # number of links added for each new node

graph = BA_model(initialnodes, m, allnodes) # Build the network of contacts

plot_network(graph)  # Draw the network

plot_degree_histogram(graph)  # Plot the degree distribution

graph = normalize_out_degree(graph)  # Normalise the edge weights with the node degree

# For verification only
print_matrix( graph )           # print the adjacent matrix
print( np.sum(graph, axis=1) )  # print the sum of the items in each column
#to verify the correctness of the  normalisation

# ---------------- Part 2 ------------

mu = 0.2  # Recovery rate
beta = 0.8  # Infection rate
max_iters = 15  # Maximum number of iterations

nodes = initial_infection(allnodes)  # Initiate the infection

plot_network_colored(graph, nodes)  # Draw the network: infected in red

for i in range(1, max_iters):
    # call SIS_step to compute the new probability prob
    # call new_state_nodes to determine the new state (i or s) of each person
    prob = SIS_step(graph, nodes, mu, beta)
    nodes = new_state_nodes(prob, nodes)
    plot_network_colored(graph, nodes)

p_vs_steps(rmse_threshold=False)
p_vs_beta(rmse_threshold=False)
p_vs_r0(rmse_threshold=False)