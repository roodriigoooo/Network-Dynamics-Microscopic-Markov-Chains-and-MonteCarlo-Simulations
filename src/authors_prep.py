"""
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