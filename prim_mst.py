from main import gnp_random_connected_graph, add_weights
import heapq


def find_adj(node, edges):
    children = []
    for i in edges:
        if i[0] == node:
            children.append((i[2], i[1], i[0]))
        elif i[1] == node:
            children.append((i[2], i[0], i[1]))
    return children


def prim_mst(nodes, edges):
    span = []
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
            span.append(curr_vertex)
            children = find_adj(curr_vertex[1], edges)
            for child in children:
                if child[1] not in visited_set:
                    heapq.heappush(frontier_minheap, child)
    return span, mst_cost


def execute():
    G = gnp_random_connected_graph(500, 1)
    nodes, edges = add_weights(G)
    tree, cost = prim_mst(nodes, edges)
    print(tree, len(tree), cost)


if __name__ == '__main__':
    execute()
