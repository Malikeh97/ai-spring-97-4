from node import *

def delete_element(l, index):
    if index == 0:
        return delete_first_element(l)
    elif index == -1 or index - len(l) == -1:
        return delete_last_element(l)
    else:
        return l[0:index] + l[index + 1:]

def delete_first_element(l):
    if len(l) == 1:
        return []
    else:
        return l[1:]

def delete_last_element(l):
    if len(l) == 1:
        return []
    else:
        return l[:-1]

def add_neighbors_to_frontier(frontier, explored, neighbors, node, maze = None, astar_heuristic = None):
    i = 0
    for n in neighbors:
        if n not in explored:
            append_permission = True
            if frontier:
                for f in frontier:
                    if n == f.cell:
                        append_permission = False
                        break
            if append_permission:
                i += 1
                if maze is None:
                    frontier.append(Node(cell = n, parent = node, path_cost = node.path_cost + 1))
                else:
                    frontier.append(Node(cell = n, parent = node, path_cost = node.path_cost + 1, h = astar_heuristic(maze, n)))
    return i

def pick_from_frontier(frontier):
    node = frontier[0]
    i, index = 0, 0
    for f in frontier:
        if f.fn < node.fn:
            node = f
            index = i
        i += 1
    return (node, index)
