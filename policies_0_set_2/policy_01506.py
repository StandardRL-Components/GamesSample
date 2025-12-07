def policy(env):
    # Strategy: Prioritize AoE blast when multiple monsters are near for efficient clears, 
    # otherwise shoot to pick off singles. Move away when threatened to avoid damage. 
    # This maximizes reward by minimizing health loss and maximizing kill speed.
    player_pos = env.player_pos
    monsters = env.monsters
    if not monsters:
        return [0, 0, 0]
    
    # Check for multiple monsters within AoE range (distance <= 2)
    count_nearby = 0
    for m in monsters:
        dx = m['pos'][0] - player_pos[0]
        dy = m['pos'][1] - player_pos[1]
        if dx*dx + dy*dy <= 4:
            count_nearby += 1
    if count_nearby >= 2:
        return [0, 0, 1]
    
    # Find nearest monster and check if threatening (distance < 1.5)
    min_dist_sq = float('inf')
    nearest_monster = None
    for m in monsters:
        dx = m['pos'][0] - player_pos[0]
        dy = m['pos'][1] - player_pos[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_monster = m
    
    if min_dist_sq < 2.25:  # 1.5^2 = 2.25
        dx = player_pos[0] - nearest_monster['pos'][0]
        dy = player_pos[1] - nearest_monster['pos'][1]
        candidate_dirs = []
        if abs(dx) > abs(dy):
            candidate_dirs.append(4 if dx > 0 else 3)
            candidate_dirs.append(2 if dy > 0 else 1)
        else:
            candidate_dirs.append(2 if dy > 0 else 1)
            candidate_dirs.append(4 if dx > 0 else 3)
        
        for move_dir in candidate_dirs:
            new_x = player_pos[0] + (1 if move_dir == 4 else -1 if move_dir == 3 else 0)
            new_y = player_pos[1] + (1 if move_dir == 2 else -1 if move_dir == 1 else 0)
            if 0 <= new_x < env.GRID_WIDTH and 0 <= new_y < env.GRID_HEIGHT:
                return [move_dir, 0, 0]
        return [0, 1, 0]
    
    return [0, 1, 0]