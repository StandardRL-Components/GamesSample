def policy(env):
    """
    Strategy: Move towards the nearest gem to maximize collection efficiency and minimize time.
    If already on a gem, use no-op to collect without time cost. Otherwise, evaluate all possible
    moves and choose the one that minimizes Manhattan distance to the nearest gem, breaking ties
    consistently (up > down > left > right) to avoid oscillation.
    """
    if env.game_over or not env.gems:
        return [0, 0, 0]
    
    player_pos = env.player_pos
    for gem in env.gems:
        if player_pos[0] == gem[0] and player_pos[1] == gem[1]:
            return [0, 0, 0]
    
    best_action = 0
    best_dist = float('inf')
    for action_idx in [1, 2, 3, 4]:
        new_x, new_y = player_pos[0], player_pos[1]
        if action_idx == 1:
            new_y = max(0, new_y - 1)
        elif action_idx == 2:
            new_y = min(env.GRID_SIZE - 1, new_y + 1)
        elif action_idx == 3:
            new_x = max(0, new_x - 1)
        elif action_idx == 4:
            new_x = min(env.GRID_SIZE - 1, new_x + 1)
        
        min_dist = float('inf')
        for gem in env.gems:
            dist = abs(new_x - gem[0]) + abs(new_y - gem[1])
            if dist < min_dist:
                min_dist = dist
        
        if min_dist < best_dist:
            best_dist = min_dist
            best_action = action_idx
    
    return [best_action, 0, 0]