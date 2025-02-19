import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc

def run_causal_discovery(X, y, alpha=0.05):
    """
    Run causal discovery using the PC algorithm from causal-learn.
    
    Parameters:
        X (pd.DataFrame): Features DataFrame (must include column 'Z')
        y (array-like): Target variable T
        alpha (float): Significance level for conditional independence tests
        
    This function combines X and y into a single dataset, runs the PC algorithm,
    prints the learned directed and undirected edges, and visualizes the resulting
    causal graph.
    """
    # Combine X and y into one DataFrame, naming the target column 'T'
    df = X.copy()
    df['T'] = y
    
    # Convert the DataFrame to a NumPy array (observations in rows, variables in columns)
    data_np = df.to_numpy()
    
    # Run the PC algorithm (using the stable variant for robustness)
    pc_result = pc(data_np, alpha=alpha, stable=True)
    
    # Print the resulting adjacency matrix (this uses the __str__ method of the graph)
    print("Learned adjacency matrix:")
    print(pc_result.G)
    
    # Extract directed and undirected edges using available methods.
    try:
        directed_edges = pc_result.G.get_directed_edges()
    except AttributeError:
        directed_edges = []
    try:
        undirected_edges = pc_result.G.get_undirected_edges()
    except AttributeError:
        undirected_edges = []
    
    print("\nDirected edges:")
    for edge in directed_edges:
        # Each edge is typically a tuple (i, j) meaning i --> j.
        print(edge)
    print("\nUndirected edges:")
    for edge in undirected_edges:
        # Each undirected edge is a tuple (i, j)
        print(edge)
    
    # Build a networkx graph for visualization.
    # Map column indices back to variable names.
    variable_names = list(df.columns)
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(variable_names)
    
    # Add directed edges
    for edge in directed_edges:
        i, j = edge
        G_nx.add_edge(variable_names[i], variable_names[j])
    
    # Add undirected edges as bidirectional edges
    for edge in undirected_edges:
        i, j = edge
        G_nx.add_edge(variable_names[i], variable_names[j])
        G_nx.add_edge(variable_names[j], variable_names[i])
    
    # Plot the causal graph
    pos = nx.spring_layout(G_nx)
    nx.draw(G_nx, pos, with_labels=True, node_size=2000, node_color="lightblue",
            font_size=10, arrows=True)
    plt.title("Causal Graph from PC Algorithm")
    plt.show()

# Example usage:
if __name__ == "__main__":
    # For demonstration, create a dummy dataset.
    # In practice, replace this with your actual data that has a column 'Z'
    np.random.seed(42)
    sample_size = 100
    X = pd.DataFrame({
        "A": np.random.randn(sample_size),
        "B": np.random.randn(sample_size),
        "Z": np.random.randn(sample_size)  # Noise column
    })
    # Suppose the target T is generated as a function of A and B plus noise.
    y = 2 * X["A"] - 3 * X["B"] + np.random.randn(sample_size)
    
    # Run the causal discovery process
    run_causal_discovery(X, y, alpha=0.05)
