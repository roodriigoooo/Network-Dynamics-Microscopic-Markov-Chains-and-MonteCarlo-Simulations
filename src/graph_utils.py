import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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