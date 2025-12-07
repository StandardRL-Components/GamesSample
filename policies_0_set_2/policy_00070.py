def policy(env):
    # Strategy: Move towards the nearest gem using Manhattan distance, avoiding traps when possible.
    # This minimizes steps to collect gems and avoids negative rewards, maximizing overall score.
    if env.game_over or not env.gem_locations:
        return [0, 0, 0]
    
    px, py = env.player_pos
    best_move = 0
    best_score = float('inf')
    
    for move in range(5):
        dx, dy = px, py
        if move == 1: dy -= 1
        elif move == 2: dy += 1
        elif move == 3: dx -= 1
        elif move == 4: dx += 1
        
        dx = max(0, min(env.GRID_WIDTH - 1, dx))
        dy = max(0, min(env.GRID_HEIGHT - 1, dy))
        new_pos = (dx, dy)
        
        if new_pos in env.trap_locations:
            continue
            
        min_dist = float('inf')
        for gem in env.gem_locations:
            dist = abs(dx - gem[0]) + abs(dy - gem[1])
            if dist < min_dist:
                min_dist = dist
                
        if min_dist < best_score:
            best_score = min_dist
            best_move = move
            
    return [best_move, 0, 0]