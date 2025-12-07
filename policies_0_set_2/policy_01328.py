def policy(env):
    """
    Navigate towards the exit while avoiding lasers. Prioritize moves that reduce Manhattan distance to exit,
    checking for wall collisions and laser intersections. Always return a valid action with no movement penalties.
    """
    robot_x, robot_y = env.robot_pos
    exit_x, exit_y = env.exit_pos
    dx = exit_x - robot_x
    dy = exit_y - robot_y
    
    # Define movement actions: [none, up, down, left, right]
    moves = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    
    # Score moves by Manhattan distance reduction, avoiding walls and lasers
    best_action = 0  # Default to no movement
    best_score = -float('inf')
    
    for action in range(5):
        if action == 0:
            new_x, new_y = robot_x, robot_y
        else:
            move_x, move_y = moves[action]
            new_x, new_y = robot_x + move_x, robot_y + move_y
        
        # Check if move is valid (not a wall and within bounds)
        if (new_x, new_y) in env.walls or not (0 <= new_x < env.GRID_WIDTH and 0 <= new_y < env.GRID_HEIGHT):
            continue
            
        # Calculate Manhattan distance reduction
        dist_reduction = abs(dx) + abs(dy) - (abs(exit_x - new_x) + abs(exit_y - new_y))
        
        # Check laser collision for the new position
        laser_collision = False
        for laser in env.lasers:
            lx, ly = laser['pos']
            angle_rad = math.radians(laser['angle'])
            # Vector from laser to candidate position
            vec_x = new_x - lx
            vec_y = new_y - ly
            # Dot product with laser direction
            dir_x = math.cos(angle_rad)
            dir_y = math.sin(angle_rad)
            dot = vec_x * dir_x + vec_y * dir_y
            # Check if candidate is in front of laser and within beam width
            if dot > 0 and abs(vec_x * dir_y - vec_y * dir_x) < 0.5:
                laser_collision = True
                break
                
        if laser_collision:
            continue
            
        # Prefer moves that reduce distance most
        if dist_reduction > best_score:
            best_score = dist_reduction
            best_action = action
            
    return [best_action, 0, 0]