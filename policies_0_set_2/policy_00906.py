def policy(env):
    """Move away from the closest circle to avoid collisions and maximize survival time."""
    if env.game_over:
        return [0, 0, 0]
    
    player_pos = env.player_pos
    min_dist_sq = float('inf')
    closest_pos = None
    for circle in env.circles:
        dx = player_pos[0] - circle['pos'][0]
        dy = player_pos[1] - circle['pos'][1]
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_pos = circle['pos']
    
    if closest_pos is None:
        return [4, 0, 0]
    
    dx = player_pos[0] - closest_pos[0]
    dy = player_pos[1] - closest_pos[1]
    
    if abs(dx) > abs(dy):
        a0 = 4 if dx > 0 else 3
    else:
        a0 = 2 if dy > 0 else 1
        
    return [a0, 0, 0]