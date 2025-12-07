def policy(env):
    """
    Prioritize collecting crystals by moving towards the nearest safe crystal (avoiding traps).
    If on a crystal, stay to collect. Otherwise, evaluate moves by Manhattan distance to nearest crystal,
    penalizing moves into traps. Break ties by action order (no-op, up, down, left, right) for consistency.
    """
    current_pos = env.player_pos
    
    # If standing on a crystal, stay to collect it
    if current_pos in env.crystal_positions:
        return [0, 0, 0]
    
    moves = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # dx,dy for each a0
    best_score = -float('inf')
    best_action = 0
    
    for action_idx, (dx, dy) in enumerate(moves):
        new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
        # Check bounds; invalid moves result in no movement
        if not (0 <= new_x < env.GRID_WIDTH and 0 <= new_y < env.GRID_HEIGHT):
            candidate_pos = current_pos
        else:
            candidate_pos = (new_x, new_y)
        
        # Avoid traps; heavily penalize moving into them
        if candidate_pos in env.trap_positions:
            score = -1000
        else:
            # Score by negative Manhattan distance to nearest crystal
            if env.crystal_positions:
                min_dist = min(abs(candidate_pos[0] - c[0]) + abs(candidate_pos[1] - c[1]) for c in env.crystal_positions)
                score = -min_dist
            else:
                score = 0  # No crystals left
        
        # Update best action if score is better
        if score > best_score:
            best_score = score
            best_action = action_idx
    
    return [best_action, 0, 0]