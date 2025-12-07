def policy(env):
    # Strategy: Prioritize collecting power-ups for invulnerability and bonus reward.
    # When no power-up is available, maximize minimum distance to monsters to avoid hits.
    # Use deterministic tie-breaking by action index order (0,1,2,3,4) for consistency.
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def move_pos(pos, a):
        x, y = pos
        if a == 1: y = max(0, y-1)
        elif a == 2: y = min(env.GRID_SIZE-1, y+1)
        elif a == 3: x = max(0, x-1)
        elif a == 4: x = min(env.GRID_SIZE-1, x+1)
        return [x, y]
    
    if env.powerup_active and env.powerup_pos is not None:
        curr_dist = manhattan(env.player_pos, env.powerup_pos)
        if curr_dist == 0:
            return [0, 0, 0]
        best_action = 0
        best_dist = curr_dist
        for a in [1,2,3,4]:
            new_pos = move_pos(env.player_pos, a)
            new_dist = manhattan(new_pos, env.powerup_pos)
            if new_dist < best_dist:
                best_dist = new_dist
                best_action = a
        return [best_action, 0, 0]
    
    best_action = 0
    best_min_dist = -1
    for a in [0,1,2,3,4]:
        new_pos = move_pos(env.player_pos, a)
        min_dist = min(manhattan(new_pos, m['pos']) for m in env.monsters)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_action = a
    return [best_action, 0, 0]