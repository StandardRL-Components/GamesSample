def policy(env):
    # Strategy: Predict trap states for the next step to avoid collisions (instant death) and minimize adjacent active traps (penalty). 
    # Prefer moves that keep the player safe, breaking ties by action index (stay, then up, down, left, right).
    if env.game_over:
        return [0, 0, 0]
    
    moves = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    current_x, current_y = env.player_pos
    active_traps_next = set()
    
    for trap in env.traps:
        if (env.steps + 1 + trap['offset']) % 3 == 0:
            active_traps_next.add((trap['pos'][0], trap['pos'][1]))
    
    best_action = 0
    best_score = -1000
    
    for action in range(5):
        dx, dy = moves[action]
        new_x = current_x + dx
        new_y = current_y + dy
        
        if not (0 <= new_x < env.GRID_WIDTH and 0 <= new_y < env.GRID_HEIGHT):
            new_x, new_y = current_x, current_y
        
        if (new_x, new_y) in active_traps_next:
            score = -1000
        else:
            count = 0
            for dx2, dy2 in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbor = (new_x + dx2, new_y + dy2)
                if neighbor in active_traps_next:
                    count += 1
            score = -count
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return [best_action, 0, 0]