def policy(env):
    """
    Strategy: Prioritize collecting gems by moving towards the closest gem while avoiding traps.
    Use Manhattan distance to evaluate moves. If multiple moves are equally good, break ties by action index (0-4).
    Secondary actions (a1, a2) are unused in this environment and set to 0.
    """
    current_pos = env.player_pos
    gems = env.gems
    traps = env.traps
    
    if not gems:
        return [0, 0, 0]
    
    actions = [0, 1, 2, 3, 4]
    dxdy = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    current_min_dist = min(abs(current_pos[0] - g[0]) + abs(current_pos[1] - g[1]) for g in gems)
    
    scores = []
    for a in actions:
        dx, dy = dxdy[a]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        if new_pos[0] < 0 or new_pos[0] >= env.GRID_WIDTH or new_pos[1] < 0 or new_pos[1] >= env.GRID_HEIGHT:
            scores.append(-10)
            continue
            
        if new_pos in traps:
            scores.append(-1000)
            continue
            
        if new_pos in gems:
            scores.append(1000)
            continue
            
        new_min_dist = min(abs(new_pos[0] - g[0]) + abs(new_pos[1] - g[1]) for g in gems)
        if new_min_dist < current_min_dist:
            scores.append(1)
        else:
            scores.append(-0.1)
            
    best_score = max(scores)
    best_actions = [i for i, s in enumerate(scores) if s == best_score]
    best_action = min(best_actions)
    
    return [best_action, 0, 0]