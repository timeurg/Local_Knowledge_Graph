import networkx as nx
import heapq

def dijkstra(graph: nx.Graph, start, end):
    queue = [(0, start, [])]
    visited = set()
    
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            
            if node == end:
                path_length = len(path) - 1
                if path_length == 0:  # Handle the case when there's only one node
                    return 1.0, path  # Return perfect similarity for single node
                return -cost / path_length, path  # Return average similarity
            
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    edge_weight = graph[node][neighbor]['weight']
                    new_cost = cost - edge_weight  # Accumulate total similarity
                    heapq.heappush(queue, (new_cost, neighbor, path))
    
    return None, None

def calculate_strongest_path(graph_data, current_step):
    G = nx.Graph()
    for node in graph_data['nodes']:
        G.add_node(node['id'])
    for edge in graph_data['edges']:
        G.add_edge(edge['from'], edge['to'], weight=edge['value'])
    
    start_node = 'Step1'
    end_node = f'Step{current_step}'
    

    try:
        avg_similarity, path = dijkstra(G, start_node, end_node)
        if path:
            if len(path) == 1:  # Handle the case when there's only one node
                return path, [], 1.0
            path_edges = list(zip(path[:-1], path[1:]))
            path_weights = [G[u][v]['weight'] for u, v in path_edges]
            return path, path_weights, avg_similarity
        else:
            return None, None, None
    except nx.NetworkXNoPath:
        return None, None, None

def serialize_graph_data(graph_data):
    serialized = {
        'nodes': graph_data['nodes'],
        'edges': [
            {
                'from': edge['from'],
                'to': edge['to'],
                'value': float(edge['value']),  # Convert float32 to regular float
                'label': f"{float(edge['value']):.2f}",  # Add similarity value as label
                'font': {'size': 10}  # Adjust font size for readability
            }
            for edge in graph_data['edges']
        ]
    }
    #print("serialized",serialized)
    return serialized