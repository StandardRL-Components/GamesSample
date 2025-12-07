def policy(env):
    """
    Strategy: Navigate snake towards the orb using Manhattan distance, avoiding self-collision and walls.
    Prioritize moves that reduce distance to orb while ensuring safety by excluding current body segments (except tail).
    Avoid reverse moves (no-ops) and break ties by preferring current direction to minimize oscillation.
    """
    head = env.snake_body[0]
    orb = env.orb_pos
    current_dir = env.direction
    safe_cells = set(env.snake_body[:-1])  # Exclude tail (safe to move into if not growing)
    
    best_action = 0  # Default to no-op (continue current direction)
    best_score = float('inf')
    
    # Evaluate each possible movement direction
    for move_code, move_vec in env.action_map.items():
        new_pos = (head[0] + move_vec[0], head[1] + move_vec[1])
        
        # Skip invalid moves: out-of-bounds, self-collision, or reverse (no-op)
        if (new_pos[0] < 0 or new_pos[0] >= env.GRID_WIDTH or
            new_pos[1] < 0 or new_pos[1] >= env.GRID_HEIGHT or
            new_pos in safe_cells or
            (move_vec[0] == -current_dir[0] and move_vec[1] == -current_dir[1])):
            continue
        
        # Score: Manhattan distance to orb (lower is better)
        score = abs(new_pos[0] - orb[0]) + abs(new_pos[1] - orb[1])
        
        # Prefer current direction to reduce oscillation
        if move_vec == current_dir:
            score -= 0.5
            
        if score < best_score:
            best_score = score
            best_action = move_code
            
    return [best_action, 0, 0]  # a1 and a2 unused in this environment