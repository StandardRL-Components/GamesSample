def policy(env):
    # Strategy: Prioritize destroying aliens while avoiding projectiles. Always fire when possible to maximize damage output.
    # Movement targets the closest alien while evading nearby threats. Secondary action unused in this shooter environment.
    player_x = env.player_pos[0]
    player_y = env.player_pos[1]
    
    # Check for immediate projectile threats
    threat = None
    min_threat_dist = float('inf')
    for proj in env.alien_projectiles:
        dist_x = abs(proj[0] - player_x)
        dist_y = player_y - proj[1]  # Positive if projectile is above player
        if 0 < dist_y < 100 and dist_x < 50:
            threat_dist = math.hypot(dist_x, dist_y)
            if threat_dist < min_threat_dist:
                threat = proj
                min_threat_dist = threat_dist
    
    if threat is not None:
        # Evade the closest threatening projectile
        if threat[0] < player_x:
            move_action = 4  # Move right
        else:
            move_action = 3  # Move left
    elif env.aliens:
        # Target the lowest (most dangerous) alien
        lowest_alien = max(env.aliens, key=lambda a: a['pos'][1])
        target_x = lowest_alien['pos'][0]
        if player_x < target_x - 10:
            move_action = 4  # Move right
        elif player_x > target_x + 10:
            move_action = 3  # Move left
        else:
            move_action = 0  # Hold position
    else:
        move_action = 0  # No movement needed
    
    # Always fire when possible (primary action)
    fire_action = 1
    
    # Secondary action unused in this environment
    secondary_action = 0
    
    return [move_action, fire_action, secondary_action]