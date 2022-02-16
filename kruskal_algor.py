"""Kruskal's algorithm"""
from main import gnp_random_connected_graph, add_weights
from prim_mst import prim_mst


def kruskal(nodes: list, edges: list) -> tuple[list[list], int]:
    """
    finds a minimum spanning forest of an undirected edge-weighted connected graph
    using Kruskal's algorithm

    :param nodes: list of graph nodes
    :param edges: list of graph edges
    :return: tuple of the minimum spanning forest and the weight of its frame
    """
    nodes, edges = list(map(lambda x: [x], nodes)), sorted(edges, key=lambda x: x[2])
    weight, edges_list = 0, []
    while edges:
        vert1, vert2, vert_weight = edges.pop(0)
        for item in nodes:
            if vert1 in item:
                index_vert1 = nodes.index(item)
            if vert2 in item:
                index_vert2 = nodes.index(item)
        if index_vert1 != index_vert2:
            nodes[index_vert1].extend(nodes.pop(index_vert2))
            weight += vert_weight
            edges_list.append([vert1, vert2])
    return edges_list, weight


if __name__ == '__main__':
    graph = add_weights(gnp_random_connected_graph(10, 0.2))
    print(kruskal(graph[0], graph[1])[1])
    print(prim_mst(graph[0], graph[1])[1])
