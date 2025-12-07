def policy(env):
    # Strategy: Avoid enemies first, then mine closest asteroid. Use vector direction to choose movement.
    # Avoid enemies within 150px by moving away. Mine asteroids within 80px. Maximizes ore collection while minimizing collision risk.
    if env.game_over:
        return [0, 0, 0]
    
    safe_dist_sq = 22500  # 150^2
    mining_range_sq = env.PLAYER_MINING_RANGE ** 2
    
    # Check for nearby enemies
    avoid_mode = False
    closest_enemy = None
    min_enemy_dist_sq = float('inf')
    for enemy in env.enemies:
        dx = env.player['x'] - enemy['x']
        dy = env.player['y'] - enemy['y']
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_enemy_dist_sq:
            min_enemy_dist_sq = dist_sq
            closest_enemy = enemy
    if min_enemy_dist_sq < safe_dist_sq:
        avoid_mode = True
    
    if avoid_mode:
        # Move away from closest enemy
        dx = env.player['x'] - closest_enemy['x']
        dy = env.player['y'] - closest_enemy['y']
        if dx == 0 and dy == 0:
            move = 0
        else:
            if abs(dx) > abs(dy):
                move = 4 if dx > 0 else 3
            else:
                move = 2 if dy > 0 else 1
        return [move, 0, 0]
    else:
        # Find closest asteroid with ore
        closest_asteroid = None
        min_ast_dist_sq = float('inf')
        for asteroid in env.asteroids:
            if asteroid['ore'] > 0:
                dx = asteroid['x'] - env.player['x']
                dy = asteroid['y'] - env.player['y']
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_ast_dist_sq:
                    min_ast_dist_sq = dist_sq
                    closest_asteroid = asteroid
        
        if closest_asteroid is None:
            return [0, 0, 0]
        
        # Move toward asteroid and mine if in range
        dx = closest_asteroid['x'] - env.player['x']
        dy = closest_asteroid['y'] - env.player['y']
        if dx == 0 and dy == 0:
            move = 0
        else:
            if abs(dx) > abs(dy):
                move = 4 if dx > 0 else 3
            else:
                move = 2 if dy > 0 else 1
        
        mine = 1 if min_ast_dist_sq < mining_range_sq else 0
        return [move, mine, 0]