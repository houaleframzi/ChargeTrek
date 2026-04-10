import networkx as nx

def find_optimal_path_bellman_ford(graph, start_node, goal_node, key='real'):
    """
    Computes the shortest path using Bellman-Ford algorithm on a MultiDiGraph,
    using only edges with the specified key ('real' or 'forecast').

    Parameters:
        graph: networkx.MultiDiGraph
        start_node: tuple (time_step, soc_level)
        goal_node: tuple (time_step, soc_level)
        key: str, either 'real' or 'forecast' to select edge type

    Returns:
        path: list of nodes from start to goal
        total_cost: float, total path cost
    """
    # Create filtered view using only edges with the specified key
    G_filtered = nx.DiGraph()
    for u, v, k, d in graph.edges(keys=True, data=True):
        if k == key:
            G_filtered.add_edge(u, v, weight=d['weight'])

    try:
        path = nx.bellman_ford_path(G_filtered, source=start_node, target=goal_node, weight='weight')
        cost = nx.bellman_ford_path_length(G_filtered, source=start_node, target=goal_node, weight='weight')
        return path, cost
    except nx.NetworkXUnbounded:
        print("Negative cycle detected.")
        return [], float('-inf')
    except nx.NetworkXNoPath:
        #print("No path found between start and goal.")
        return [], float('inf')