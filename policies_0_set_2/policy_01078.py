def policy(env):
    """
    Strategy: Prioritize collecting gems while avoiding enemies. For each possible movement,
    evaluate the immediate reward (gem collection) and movement rewards (distance changes to
    closest gem and enemy). Choose the action that maximizes immediate gem collection if safe,
    otherwise minimizes distance to gems while maximizing distance from enemies.
    """
    player_pos = env.player_pos
    gems = env.gem_positions
    enemies = [e['pos'] for e in env.enemies]
    GRID_COLS, GRID_ROWS = env.GRID_COLS, env.GRID_ROWS
    
    # Calculate current distances
    current_dist_gem = min([abs(player_pos[0]-g[0]) + abs(player_pos[1]-g[1]) for g in gems]) if gems else 0
    current_dist_enemy = min([abs(player_pos[0]-e[0]) + abs(player_pos[1]-e[1]) for e in enemies]) if enemies else float('inf')
    
    best_action = 0
    best_score = -float('inf')
    
    for action in range(5):
        dx, dy = 0, 0
        if action == 1: dy = -1
        elif action == 2: dy = 1
        elif action == 3: dx = -1
        elif action == 4: dx = 1
        
        new_x = max(0, min(GRID_COLS-1, player_pos[0] + dx))
        new_y = max(0, min(GRID_ROWS-1, player_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        # Check for immediate gem collection
        collect_reward = 10.0 if new_pos in gems else 0.0
        # Check for enemy collision penalty
        hit_penalty = -30.0 if new_pos in enemies else 0.0
        
        # Calculate distance changes
        new_dist_gem = min([abs(new_x-g[0]) + abs(new_y-g[1]) for g in gems]) if gems else 0
        new_dist_enemy = min([abs(new_x-e[0]) + abs(new_y-e[1]) for e in enemies]) if enemies else float('inf')
        
        # Movement rewards (environment uses closest gem/enemy)
        move_reward_gem = (current_dist_gem - new_dist_gem) * 1.0
        move_reward_enemy = (current_dist_enemy - new_dist_enemy) * (-1.0)  # Negative for moving away
        
        total_score = collect_reward + hit_penalty + move_reward_gem + move_reward_enemy
        
        if total_score > best_score:
            best_score = total_score
            best_action = action
            
    return [best_action, 0, 0]