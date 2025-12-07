def policy(env):
    # Strategy: Move towards the nearest gem using Manhattan distance, avoiding obstacles and walls.
    # Prioritize immediate gem collection when adjacent. Use deterministic tie-breaking to avoid oscillation.
    actions = [1, 2, 3, 4, 0]  # up, down, left, right, none
    best_action = 0
    best_dist = float('inf')
    py, px = env.player_pos
    
    for move in actions:
        if move == 0:
            new_pos = (py, px)
        else:
            dy, dx = {1: (-1,0), 2: (1,0), 3: (0,-1), 4: (0,1)}[move]
            new_pos = (py + dy, px + dx)
        
        # Skip invalid moves (walls or obstacles)
        if (new_pos[0] < 0 or new_pos[0] >= env.maze.shape[0] or 
            new_pos[1] < 0 or new_pos[1] >= env.maze.shape[1] or
            env.maze[new_pos] != 0 or new_pos in env.obstacles):
            continue
            
        # Calculate min Manhattan distance to gems from new position
        min_gem_dist = min((abs(new_pos[0]-gy) + abs(new_pos[1]-gx) for gy, gx in env.gems), default=0)
        if min_gem_dist < best_dist:
            best_dist = min_gem_dist
            best_action = move
            
    return [best_action, 0, 0]