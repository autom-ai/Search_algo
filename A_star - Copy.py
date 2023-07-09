from operator import itemgetter
import numpy as np
import path_planning as pp
import heuristics

def search(_map, start, goal, metric='euclidean', info=None):
    # Finds the neighbors of the given node.
    def get_neighbors(node):
        actions = [[-1, 0],  # go up
                   [0, -1],  # go left
                   [1, 0],  # go down
                   [0, 1]]  # do right

        possible_actions = []
        visited_states = [element for element in came_from.values()]
        in_frontier = [node[3:] for node in frontier]

        # Find out the possible actions to take from the current node.
        for action in actions:
            neighbor_node = [node[0] + action[0], node[1] + action[1]]
            
            # -1 not in neighbor_node - if the neighbor node is outside the map.
            # neighbor_node[0] <= len(_map) - if the neighbor node is outside the map.
            # neighbor_node[1] <= len(_map[0]) - if the neighbor node is outside the map.
            if -1 not in neighbor_node and neighbor_node[0] < len(_map) and neighbor_node[1] < len(_map[0]):
                # neighbor_node not in visited_states - see if we have been at this node before
                # neighbor_node not in in_frontier - see if this node has already been discovered (is in frontier list)
                # _map[neighbor_node[0]][neighbor_node[1]] != -1 - check if the next state is not an obstacle
                if neighbor_node not in visited_states and neighbor_node not in in_frontier and _map[neighbor_node[0]][neighbor_node[1]] != -1:
                    possible_actions.append(neighbor_node)

        return (possible_actions)

    
    # Computes the cost to reach the next cell.
    def cost_function(g):
        return g + moving_cost

    # Calculate heuristic values of the map.
    heuristic_map = np.array([])
    if metric == 'euclidean':
        print('Calculating euclidean distance map...')
        heuristic_map = heuristics.euclidean(_map, goal)
    elif metric == 'manhattan':
        print('Calculating manhattan distance map...')
        heuristic_map = heuristics.manhattan(_map, goal)
    elif metric == 'special':
        print('Calculating special distance map...')
        heuristic_map = heuristics.special(_map, start, goal, info)
    
    # cost moving to another cell
    moving_cost = 1

    # open list
    frontier = []

    start = start.tolist()
    goal = goal.tolist()

    # add starting cell to open list
    frontier.append([heuristic_map[start[0],start[1]] + 0, 0, heuristic_map[start[0],start[1]]] + start)   # [f, g, h, x, y], f = g + h

    # path taken
    came_from = {}

    # expanded list with cost value for each cell
    cost = {}
    visited_nodes = [[],[]]

    g = 0
    exp_nodes = 0
    
    solved_map = np.copy(_map)
    
    print('Searching...')

    # If there is still nodes to open.
    while frontier:
        # Get the state which has the lowest cost.
        current_node = frontier.pop(0)
        visited_nodes[0].append(current_node[3])
        visited_nodes[1].append(current_node[4])
        exp_nodes += 1

        # check if the goal is reached
        if current_node[3:] == goal:
            break

        # for each neighbor of the current cell
        # Implement get_neighbors function (return nodes to expand next)
        # (make sure you avoid repetitions!)
        for next in get_neighbors(current_node[3:]):

            # compute cost to reach next cell
            # Implement cost function
            cost = cost_function(g)

            # add next cell to open list
            frontier.append([cost+heuristic_map[next[0], next[1]], cost, heuristic_map[next[0], next[1]], next[0], next[1]])
            frontier = sorted(frontier, key=itemgetter(0))
            # add to path
            came_from[tuple(next)] = current_node[3:]

        if solved_map[current_node[3], current_node[4]] != -2:
            solved_map[current_node[3], current_node[4]] = g
        
        g += moving_cost

    return pp.solve_path(came_from, start, goal), solved_map, exp_nodes
