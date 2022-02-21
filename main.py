import random
import time
import pandas as pd
import networkx as nx
from kruskal_algor import kruskal
from prim_mst import prim_mst
from itertools import combinations, groupby


def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: float) -> list[tuple[int, int]]:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)

    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.randint(0, 20)
    return G


def add_weights(graph):
    """
    Inherently graph is unweighted
    Add weights to the graph
    :param graph:
    :return: tuple(list_of_edges, list_of_nodes)
    """
    edges = list(map(lambda x: (x[0], x[1], x[2]['weight']), graph.edges(data=True)))
    nodes = list(graph.nodes)
    return nodes, edges


def results(vertex):
    """
    set timer and make a research
    :param vertex: int
    :return: list
    """
    result = []
    for possibility in range(1, 101, 1):
        possibility = possibility/100
        nodes, edges = add_weights(gnp_random_connected_graph(vertex, possibility))
        start = time.time()
        kruskal(nodes, edges)
        mid = time.time()
        prim_mst(nodes, edges)
        end = time.time()
        result.append([round(float(mid-start), 5), round(float(end-mid), 5), int(vertex), possibility])
    return result


def saving_results(start, end, step, repetition):
    """
    save result of results (func)
    start (vertex), end (vertex)
    :param start: int
    :param end: int
    :param step: int
    :param repetition: int
    :return: None
    """
    result = []
    for vertex in range(start, end, step):
        print(vertex)
        for _ in range(repetition):
            result.extend(results(vertex))
    file = pd.DataFrame(result, columns=['Kruskal_time', 'Prim_time', 'Num_of_nodes', 'Completeness'])
    file.to_csv('dataset.csv', mode='a')


def sort_csv():
    """
    sort csv file by third and fourth columns
    :return: None
    """
    file = pd.read_csv('dataset.csv')
    result_dict = []
    for index, row in file.iterrows():
        row = list(row)
        if row[1] == 'Kruskal_time' or row[1]>16 or row[2]>16:
            continue
        row = [float(row[1]), float(row[2]), int(row[3]), float(row[4])]
        result_dict.append(row)
    result = sorted(result_dict, key=lambda x: (x[2], x[3]), reverse=True)
    file = pd.DataFrame(result, columns=['Kruskal_time', 'Prim_time', 'Num_of_nodes', 'Completeness'])
    file.to_csv('dataset.csv')
