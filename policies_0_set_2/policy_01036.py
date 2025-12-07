def policy(env):
    # Prioritize safety by avoiding enemies, then move toward nearest gem to maximize reward collection.
    # Uses Manhattan distance to evaluate moves: high weight on enemy avoidance, then gem proximity.
    player_x, player_y = env.player_pos
    active_gems = [g for g in env.gems if g['respawn_timer'] == 0]
    enemy_positions = [e['pos'] for e in env.enemies]
    
    best_score = -10**9
    best_move = 0
    
    for move in [0, 1, 2, 3, 4]:
        dx, dy = 0, 0
        if move == 1: dy = -1
        elif move == 2: dy = 1
        elif move == 3: dx = -1
        elif move == 4: dx = 1
        
        new_x = max(0, min(env.GRID_WIDTH-1, player_x + dx))
        new_y = max(0, min(env.GRID_HEIGHT-1, player_y + dy))
        
        min_enemy_dist = min(abs(new_x - ex) + abs(new_y - ey) for ex, ey in enemy_positions)
        min_gem_dist = min(abs(new_x - gx) + abs(new_y - gy) for g in active_gems for gx, gy in [g['pos']]) if active_gems else 0
        
        if min_enemy_dist == 0:
            score = -100000
        else:
            score = min_enemy_dist * 100 - min_gem_dist
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return [best_move, 0, 0]