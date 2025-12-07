def policy(env):
    # Strategy: Prioritize collecting gems while avoiding enemies. For each possible movement, compute a score
    # based on gem proximity (higher for moving closer) and enemy avoidance (penalize moving closer to enemies).
    # Break ties by preferring movement that collects gems or increases distance from enemies.
    current_pos = env.player['pos']
    current_x, current_y = current_pos[0], current_pos[1]
    
    gems = [(gem['pos'][0], gem['pos'][1]) for gem in env.gems]
    enemies = [(enemy['pos'][0], enemy['pos'][1]) for enemy in env.enemies]
    
    if not gems:
        return [0, 0, 0]
    
    best_action = 0
    best_score = -10**9
    
    for action in range(5):
        dx, dy = 0, 0
        if action == 1: dy = -1
        elif action == 2: dy = 1
        elif action == 3: dx = -1
        elif action == 4: dx = 1
        
        new_x = max(0, min(env.GRID_WIDTH - 1, current_x + dx))
        new_y = max(0, min(env.GRID_HEIGHT - 1, current_y + dy))
        new_pos = (new_x, new_y)
        
        if new_pos in enemies:
            score = -1000
        else:
            current_gem_dist = min((current_x - gx)**2 + (current_y - gy)**2 for gx, gy in gems)
            new_gem_dist = min((new_x - gx)**2 + (new_y - gy)**2 for gx, gy in gems)
            gem_score = current_gem_dist - new_gem_dist
            
            enemy_score = 0
            if enemies:
                current_enemy_dist = min((current_x - ex)**2 + (current_y - ey)**2 for ex, ey in enemies)
                new_enemy_dist = min((new_x - ex)**2 + (new_y - ey)**2 for ex, ey in enemies)
                enemy_score = new_enemy_dist - current_enemy_dist
                
            score = gem_score + 2 * enemy_score
            if new_pos in gems:
                score += 10
                
        if score > best_score:
            best_score = score
            best_action = action
            
    return [best_action, 0, 0]