from main import gnp_random_connected_graph, add_weights
import heapq


def find_adj(node, edges):
    children = []
    for i in edges:
        if i[0] == node:
            children.append((i[2], i[1]))
        elif i[1] == node:
            children.append((i[2], i[0]))
    return children


def prim_mst(nodes, edges):
    mst_cost = 0
    visited_set = set()
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
            children = find_adj(curr_vertex[1], edges)
            for child in children:
                if child[1] not in visited_set:
                    heapq.heappush(frontier_minheap, child)
    return mst_cost


def execute():
    G = gnp_random_connected_graph(5, 1)
    nodes, edges = add_weights(G)
    print(nodes, edges)
    print(prim_mst(nodes, edges))


if __name__ == '__main__':
    execute()
