def policy(env):
    # Strategy: Prioritize collecting gems by moving toward nearest gem and using collect when in range.
    # Avoid enemies by moving away when too close. Use boost only when escaping immediate danger.
    # This balances gem collection rewards with health preservation for maximum long-term reward.
    
    # Get current state
    player_pos = env.player_pos
    gems = env.gems
    enemies = env.enemies
    
    # Find nearest gem
    min_gem_dist = float('inf')
    nearest_gem = None
    for gem in gems:
        dist = math.hypot(player_pos[0]-gem[0], player_pos[1]-gem[1])
        if dist < min_gem_dist:
            min_gem_dist = dist
            nearest_gem = gem
    
    # Find nearest enemy and distance
    min_enemy_dist = float('inf')
    nearest_enemy = None
    for enemy in enemies:
        dist = math.hypot(player_pos[0]-enemy['pos'][0], player_pos[1]-enemy['pos'][1])
        if dist < min_enemy_dist:
            min_enemy_dist = dist
            nearest_enemy = enemy['pos']
    
    # Determine movement direction
    dx, dy = 0, 0
    a2 = 0  # Default no boost
    
    if min_enemy_dist < 50:  # Enemy is close - prioritize evasion
        dx = player_pos[0] - nearest_enemy[0]
        dy = player_pos[1] - nearest_enemy[1]
        if min_enemy_dist < 30:  # Very close - use boost to escape
            a2 = 1
    elif nearest_gem:  # Move toward nearest gem
        dx = nearest_gem[0] - player_pos[0]
        dy = nearest_gem[1] - player_pos[1]
    
    # Convert direction to discrete movement
    if abs(dx) > abs(dy):
        a0 = 4 if dx > 0 else 3  # Right or left
    else:
        a0 = 2 if dy > 0 else 1  # Down or up
    
    # Use collect action when near any gem
    a1 = 1 if min_gem_dist < env.GEM_COLLECT_RADIUS else 0
    
    return [a0, a1, a2]