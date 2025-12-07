def policy(env):
    # Greedy Manhattan distance minimization to nearest crystal for immediate collection rewards.
    # Avoids no-ops by checking bounds, prioritizes moves that reduce distance, and breaks ties by direction order.
    current_pos = env.player_pos
    # Check if currently on a crystal (collect by staying)
    for crystal in env.crystals:
        if current_pos[0] == crystal['pos'][0] and current_pos[1] == crystal['pos'][1]:
            return [0, 0, 0]
    
    crystals_list = [c['pos'] for c in env.crystals]
    # Calculate current min Manhattan distance to any crystal
    current_min_dist = float('inf')
    for c in crystals_list:
        dist = abs(current_pos[0] - c[0]) + abs(current_pos[1] - c[1])
        if dist < current_min_dist:
            current_min_dist = dist
            
    moves = [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))]
    best_move = 0
    best_dist = current_min_dist
    # First pass: find move that strictly reduces distance
    for code, delta in moves:
        new_x = current_pos[0] + delta[0]
        new_y = current_pos[1] + delta[1]
        if 0 <= new_x < env.GRID_WIDTH and 0 <= new_y < env.GRID_HEIGHT:
            min_dist = float('inf')
            for c in crystals_list:
                dist = abs(new_x - c[0]) + abs(new_y - c[1])
                if dist < min_dist:
                    min_dist = dist
            if min_dist < best_dist:
                best_dist = min_dist
                best_move = code
                
    if best_move != 0:
        return [best_move, 0, 0]
        
    # Second pass: if no strict improvement, choose valid move with minimal distance (may be equal)
    best_move = 0
    best_dist = float('inf')
    for code, delta in moves:
        new_x = current_pos[0] + delta[0]
        new_y = current_pos[1] + delta[1]
        if 0 <= new_x < env.GRID_WIDTH and 0 <= new_y < env.GRID_HEIGHT:
            min_dist = float('inf')
            for c in crystals_list:
                dist = abs(new_x - c[0]) + abs(new_y - c[1])
                if dist < min_dist:
                    min_dist = dist
            if min_dist < best_dist:
                best_dist = min_dist
                best_move = code
                
    return [best_move, 0, 0]