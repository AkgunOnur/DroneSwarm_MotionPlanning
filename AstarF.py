class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, action=None):
        self.parent = parent
        self.position = position
        self.action = action

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        for pos1, pos2 in zip(self.position, other.position):
            if pos1 != pos2:
                return False
        return True
def astar_drone(start, end, env):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    grid_res = env.grid_res
    x_lim, y_lim, z_lim = env.x_lim, env.y_lim, env.z_lim
    obs_x0, obs_y0, obs_z0 = env.obstacle_points[0][0], env.obstacle_points[0][1], env.obstacle_points[0][2]  
    obs_x1, obs_y1, obs_z1 = env.obstacle_points[0][3], env.obstacle_points[0][4], env.obstacle_points[0][5]  

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append([current.position, current.action])
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for index, new_position in enumerate([(grid_res, 0, 0), (-grid_res, 0, 0), (0, grid_res, 0), (0, -grid_res, 0)]): # Adjacent squares
#         for new_position in [(0, -grid_res, 0), (0, grid_res, 0), (-grid_res, 0, 0), (grid_res, 0, 0)]: # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1], current_node.position[2] + new_position[2])
#             print ("Node position: ", node_position)
#             print ("Node index: {0} Node pos: {1}/{2}".format(node_index, uncertainty_grids[node_index], node_position))
            
            # Make sure within range
            if node_position[0] > x_lim or node_position[0] < -x_lim or node_position[1] > y_lim or node_position[1] < -y_lim or node_position[2] > z_lim or node_position[2] < 0:
#                 print ("It's not within the range. Node position: ", node_position)
                continue

            node_index = env.get_closest_grid(node_position)
              
            # Make sure flyable area
            if (node_position[0] >= obs_x0 and node_position[0] <= obs_x1) and (node_position[1] >= obs_y0 and node_position[1] <= obs_y1):
                print ("It's a obstacle place. Node position: ", node_position)
                continue
    
            
#             if node_index in env.agent_pos_index:
# #                 print ("Another agent is here. Node position: ", node_position)
#                 continue
                    
            # if node_index in env.obstacle_indices:
            #     # print ("It's not a flyable area. Node position: ", node_position)
            #     continue
                

            # Create new node
            new_node = Node(current_node, node_position, index)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue
            # Add the child to the open list
            open_list.append(child)


def astar_drone_2(start, end, env):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    grid_res = env.grid_res
    x_lim, y_lim, z_lim = env.x_lim, env.y_lim, env.z_lim
    obs_x0, obs_y0, obs_z0 = env.obstacle_points[0][0], env.obstacle_points[0][1], env.obstacle_points[0][2]  
    obs_x1, obs_y1, obs_z1 = env.obstacle_points[0][3], env.obstacle_points[0][4], env.obstacle_points[0][5]  


    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = open_list[0]
        print ("current_node: ", current_node)
        print ("end_node: ", end_node)
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append([current.position,current.position])
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -grid_res, 0), (0, grid_res, 0), (-grid_res, 0, 0), (grid_res, 0, 0), (0, 0, grid_res), (0, 0, -grid_res)]: # Adjacent squares
#         for new_position in [(0, -grid_res, 0), (0, grid_res, 0), (-grid_res, 0, 0), (grid_res, 0, 0)]: # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1], current_node.position[2] + new_position[2])
            node_index = env.get_closest_grid(node_position)
#             print ("Node position: ", node_position)
            
            # Make sure within range
            if node_position[0] > x_lim or node_position[0] < -x_lim or node_position[1] > y_lim or node_position[1] < -y_lim or node_position[2] > z_lim or node_position[2] < 0:
#                 print ("It's not within the range. Node position: ", node_position)
                continue
              
            # Make sure flyable area
            if node_position[0] >= obs_x0 and node_position[0] <= obs_x1 and \
                node_position[1] >= obs_y0 and node_position[1] <= obs_y1 and \
                node_position[2] >= obs_z0 and node_position[2] <= obs_z1:
#                     print ("It's a obstacle place. Node position: ", node_position)
                    continue
    
            
            if node_index in env.agent_pos_index:
#                 print ("Another agent is here. Node position: ", node_position)
                continue
                    
#             if node_position in obstacle_positions or node_position in agents_positions:
#                 print ("It's not a flyable area. Node position: ", node_position)
#                 continue
                

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)