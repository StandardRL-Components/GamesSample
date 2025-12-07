def policy(env):
    # This policy maximizes reward by prioritizing exit proximity, gold collection, and enemy avoidance.
    # It evaluates each movement direction based on rewards, distance change, and health risks.
    # a1 and a2 are set to 0 since the environment only uses movement (a0).
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    enemies = env.enemies
    golds = env.golds
    health = env.player_health
    
    best_action = 0
    best_score = -float('inf')
    movements = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
    
    for idx, (dx, dy) in enumerate(movements):
        new_x, new_y = player_pos[0] + dx, player_pos[1] + dy
        if idx != 0 and not (0 <= new_x < env.GRID_SIZE and 0 <= new_y < env.GRID_SIZE):
            continue
            
        score = 0.0
        candidate_pos = [new_x, new_y]
        
        if candidate_pos == exit_pos:
            score += 1000.0
        if candidate_pos in enemies:
            score += 1.0
        if candidate_pos in golds:
            score += 0.5
            
        curr_dist = abs(player_pos[0]-exit_pos[0]) + abs(player_pos[1]-exit_pos[1])
        new_dist = abs(new_x-exit_pos[0]) + abs(new_y-exit_pos[1])
        score += (curr_dist - new_dist) * 0.1
        
        adjacent_enemies = 0
        for enemy in enemies:
            if enemy == candidate_pos:
                continue
            if abs(new_x - enemy[0]) + abs(new_y - enemy[1]) == 1:
                adjacent_enemies += 1
        if adjacent_enemies > 0:
            score -= (1.0 / (health + 0.1)) * adjacent_enemies
            
        if score > best_score:
            best_score = score
            best_action = idx
            
    return [best_action, 0, 0]