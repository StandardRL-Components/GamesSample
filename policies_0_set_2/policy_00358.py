def policy(env):
    # Strategy: Connect adjacent nodes of the same color to complete groups efficiently.
    # Prioritize moves that complete groups for maximum reward, using Manhattan distance
    # to navigate to target nodes. Avoid invalid connections to prevent penalty.
    current_pos = tuple(env.cursor_pos)
    
    # If a node is selected, try to connect to valid adjacent node
    if env.selected_node is not None:
        sel_color = env.nodes[env.selected_node]
        # Check adjacent cells for valid connections
        for dx, dy, move_dir in [(0,-1,1), (0,1,2), (-1,0,3), (1,0,4)]:
            adj_pos = (env.selected_node[0] + dx, env.selected_node[1] + dy)
            if (adj_pos in env.nodes and env.nodes[adj_pos] == sel_color and
                tuple(sorted((env.selected_node, adj_pos))) not in env.connections):
                # Move toward valid adjacent node if not already there
                if current_pos != adj_pos:
                    return [move_dir, 0, 0]
                else:
                    return [0, 1, 0]  # Connect when reached
        # No valid adjacent connections - cancel selection
        return [0, 0, 1]
    
    # Find incomplete color groups and target closest node
    target_node = None
    min_dist = float('inf')
    for color_idx, nodes in env.color_node_map.items():
        if not env._is_group_connected(color_idx):
            for node in nodes:
                dist = abs(node[0] - current_pos[0]) + abs(node[1] - current_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target_node = node
    
    # Move toward target node or select if reached
    if target_node is not None:
        if current_pos == target_node:
            return [0, 1, 0]  # Select node
        # Calculate movement direction
        dx = target_node[0] - current_pos[0]
        dy = target_node[1] - current_pos[1]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    
    # Default: no movement
    return [0, 0, 0]