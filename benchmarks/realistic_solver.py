import networkx as nx
from .magic_solver import find_optimal_path_bellman_ford

def run_stepwise_replanning_old(graph, start_node, goal_time, goal_soc, key_forecast="forecast", key_real="real"):
    """
    Performs step-by-step path replanning. Initially plans with forecast prices, then updates the graph at each time step
    with real prices and re-computes the optimal path.

    Parameters:
        graph: networkx.MultiDiGraph containing both 'forecast' and 'real' weights
        start_node: tuple (start_time, start_soc)
        goal_time: int, final time step (usually 95)
        goal_soc: int, desired SoC level at departure

    Returns:
        trajectory: list of nodes visited step by step
        total_cost: cumulative cost from forecast and real transitions
    """
    current_node = start_node
    trajectory = [current_node]
    total_cost = 0

    while current_node[0] < goal_time:
        # Use forecast to plan ahead
        goal_node = (goal_time, goal_soc)
        path, _ = find_optimal_path_bellman_ford(graph, current_node, goal_node, key=key_forecast)
        
        if len(path) < 2:
            #print("Path terminated early.")
            break

        next_node = path[1]

        # Take first step using forecast plan
        if graph.has_edge(current_node, next_node, key=key_real):
            edge_data = graph.get_edge_data(current_node, next_node)[key_real]
        else:
            edge_data = graph.get_edge_data(current_node, next_node)[key_forecast]

        cost = edge_data["weight"]
        total_cost += cost
        trajectory.append(next_node)
        current_node = next_node

    return trajectory, total_cost


def run_stepwise_replanning(graph, start_node, goal_time, goal_soc,
                                           key_forecast="forecast", key_real="real"):
    current_node = start_node
    trajectory = [current_node]
    total_cost = 0

    while current_node[0] < goal_time:
        # Build a filtered forecast graph
        G_step = nx.DiGraph()
        for u, v, k, d in graph.edges(keys=True, data=True):
            if k == key_forecast:
                G_step.add_edge(u, v, weight=d['weight'])

        # But overwrite the edge from current_node with real weight if available
        for v in graph.successors(current_node):
            if graph.has_edge(current_node, v, key=key_real):
                real_weight = graph.get_edge_data(current_node, v)[key_real]['weight']
                G_step[current_node][v]['weight'] = real_weight

        # Plan with updated weights
        goal_node = (goal_time, goal_soc)
        try:
            path = nx.bellman_ford_path(G_step, source=current_node, target=goal_node, weight='weight')
        except:
            break

        if len(path) < 2:
            break

        next_node = path[1]
        edge_data = graph.get_edge_data(current_node, next_node)
        cost = edge_data.get(key_real, edge_data[key_forecast])["weight"]

        total_cost += cost
        trajectory.append(next_node)
        current_node = next_node

    return trajectory, total_cost
