import numpy as np

from src.authors_prep import print_matrix, plot_network
from src.graph_utils import BA_model, normalize_out_degree, plot_degree_histogram
from src.evolution_utils import (
    plot_network_colored,
    initial_infection,
    SIS_step,
    new_state_nodes,
    p_vs_steps,
    p_vs_beta,
    p_vs_r0
)

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