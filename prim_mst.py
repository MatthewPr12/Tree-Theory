"""
Prim's algorithm
"""
import heapq


def find_adj(node, edges):
    """
    find nodes adjacent to the given
    :param node:
    :param edges:
    :return: list[tuples] -> adjacent nodes
    """
    children = []
    for i in edges:
        if i[0] == node:
            children.append((i[2], i[1], i[0]))
        elif i[1] == node:
            children.append((i[2], i[0], i[1]))
    return children


def prim_mst(nodes, edges):
    """
    perform Prim's algorithm
    :param nodes:
    :param edges:
    :return: spanning tree and cost of it
    """
    span, mst_cost, visited_set = [], 0, set()
    start_vertex = nodes[0]
    frontier_minheap = find_adj(start_vertex, edges)
    heapq.heapify(frontier_minheap)
    visited_set.add(start_vertex)
    while len(visited_set) < len(nodes):
        curr_vertex = heapq.heappop(frontier_minheap)
        cost = curr_vertex[0]
        if curr_vertex[1] not in visited_set:
            mst_cost += cost
            visited_set.add(curr_vertex[1])
            span.append(curr_vertex)
            children = find_adj(curr_vertex[1], edges)
            for child in children:
                if child[1] not in visited_set:
                    heapq.heappush(frontier_minheap, child)
    return span, mst_cost
