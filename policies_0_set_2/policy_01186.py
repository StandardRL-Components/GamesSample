def policy(env):
    # Strategy: Move directly toward the current number to maximize collection rate while maintaining safe distance from enemies.
    # Avoids enemies when within threshold distance, otherwise prioritizes shortest path to number.
    if env.game_over:
        return [0, 0, 0]
    
    p_x, p_y = env.ninja_pos
    n_x, n_y = env.number['pos']
    dx = n_x - p_x
    dy = n_y - p_y
    
    enemy_threshold_sq = 2500  # 50px squared
    closest_enemy_dist_sq = float('inf')
    avoid_x, avoid_y = 0, 0
    
    for enemy in env.enemies:
        e_x, e_y = enemy['pos']
        edx = e_x - p_x
        edy = e_y - p_y
        dist_sq = edx*edx + edy*edy
        if dist_sq < closest_enemy_dist_sq:
            closest_enemy_dist_sq = dist_sq
            avoid_x, avoid_y = -edx, -edy
    
    if closest_enemy_dist_sq < enemy_threshold_sq:
        dx, dy = avoid_x, avoid_y
    
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
    
    return [movement, 0, 0]