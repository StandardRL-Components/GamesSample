def policy(env):
    # Strategy: Maximize score by targeting the most valuable match (size * cascade potential) within minimal moves. 
    # Prioritize immediate matches at cursor, then navigate to largest nearby group considering Manhattan distance.
    if env.game_state != "IDLE" or env.game_over:
        return [0, 0, 0]
    
    x, y = env.cursor_pos
    current_group = env._find_adjacent_group(x, y)
    if len(current_group) >= 3:
        return [0, 1, 0]
    
    all_matches = env._find_all_matches()
    if not all_matches:
        return [0, 0, 0]
    
    best_score = -1
    target_pos = None
    for group in all_matches:
        group_size = len(group)
        min_dist = float('inf')
        for (r, c) in group:
            dist = abs(c - x) + abs(r - y)
            if dist < min_dist:
                min_dist = dist
        score = group_size * 10 - min_dist
        if score > best_score:
            best_score = score
            target_pos = (r, c)
    
    if target_pos is None:
        return [0, 0, 0]
    
    tr, tc = target_pos
    dx = tc - x
    dy = tr - y
    
    if abs(dx) > abs(dy):
        action0 = 4 if dx > 0 else 3
    else:
        action0 = 2 if dy > 0 else 1 if dy < 0 else 0
    
    return [action0, 0, 0]