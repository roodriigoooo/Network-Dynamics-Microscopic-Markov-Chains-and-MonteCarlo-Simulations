# Network-Dynamics-Microscopic-Markov-Chains-and-MonteCarlo-Simulations
This Python program is designed to simulate and visualize the dynamics of infectious diseases spreading through a network using the SIS (Susceptible-Infected-Susceptible) model. It incorporates the Barabási-Albert model to generate networks that mimic the complex, scale-free nature of real-world social connections. 

Overview
This project provides a comprehensive toolset for simulating the spread of infectious diseases through a population modeled as a network. The core of the simulation utilizes the Susceptible-Infected-Susceptible (SIS) model of disease spread, where each individual in the network can either be susceptible or infected at any given time. The network's structure is generated using the Barabási-Albert (BA) model, which is known for its ability to create scale-free networks that resemble the uneven connectivity patterns seen in real-world social networks.

Features
Network Generation: Start with a fully connected mesh and iteratively add nodes using preferential attachment, creating a scale-free network.
Disease Simulation: Implement the SIS model to simulate the spread of an infectious disease through the network. The program allows adjusting parameters like the infection rate (beta) and recovery rate (mu).
Visualization: Visualize the network with nodes color-coded based on their infection status and view the degree distribution of the nodes.
Probability Calculations: Calculate the probabilities of infection spread within the network, taking into account individual node connectivity.
Dynamic Updates: Continuously update the state of the network and the nodes based on disease spread dynamics over multiple iterations.
Statistical Analysis: Generate and plot data showing the relationship between the infection ratio and various parameters such as the initial number of infected individuals, beta, and R0 values across different simulations.

Dependencies
NumPy: For efficient numerical computations.
NetworkX: To manage complex network data structures.
Matplotlib: For creating static, interactive, and animated visualizations in Python.
Colorama: To enhance terminal outputs for better clarity and visualization.

Installation
Clone this repository to your local machine using:

git clone [repository-url]

Ensure you have Python installed and proceed to install the required libraries:

pip install numpy networkx matplotlib colorama

To run the simulation, navigate to the cloned directory and execute the script:

python [script_name].py

You can modify parameters like mu, beta, the initial number of infected nodes, and the total number of nodes to see how they affect the disease spread.

Upon running the script, various visual outputs will be shown, such as:
Network diagrams showing the spread of infection over time.
Degree distribution of the network.
Plots showing the evolution of the infection ratio over simulation steps.
