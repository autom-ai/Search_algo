import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski
# Returns map with euclidean heuristic values.
def euclidean(_map, goal):
    heuristic_map = np.copy(_map)
    for ix, iy in np.ndindex(_map.shape):
        heuristic_map[ix, iy] = np.linalg.norm(np.array([ix, iy]) - goal)
    return heuristic_map



# Returns map with manhattan heuristic values.
def manhattan(_map, goal):
    heuristic_map = np.copy(_map)
    for ix, iy in np.ndindex(_map.shape):
        heuristic_map[ix, iy] = minkowski(np.array([ix, iy]), goal, p=1)
    return heuristic_map


# Returns map with special heuristic values.
def special(_map, start, goal, info):
    heuristic_map = np.copy(_map)
    for ix, iy in np.ndindex(_map.shape):
        heuristic_map[ix,iy] = np.linalg.norm(np.array([ix,iy]) - goal)
        
    
    # If starting position is on the upper side of the map. (Since the y-axis is inverted)
    if start[0] > info[0]/2:
        # Make a vertical line of 0's below the starting position, encouraging the agent to move down.
        heuristic_map[start[1], start[0]:info[0]] = 0
        heuristic_map[info[0]:, :] = 0
        
    
    # If the starting position is on the lower side of the map. (Since the y-axis is inverted)
    elif start[0] < info[0]/2:
        # Make a vertical line of 0's above the starting position, encouraging the agent to move up.
        heuristic_map[start[1], info[1]:start[0]] = 0
        heuristic_map[:info[1], :] = 0
        
        
    return heuristic_map