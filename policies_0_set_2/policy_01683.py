def policy(env):
    # Strategy: Prioritize survival by dodging projectiles, then target closest alien for maximum reward.
    # Avoid firing when no aliens present to prevent penalty. Always fire when cooldown allows and aliens exist.
    player_x = env.player_pos.x
    player_y = env.player_pos.y
    
    # Check for immediate projectile threats
    threat_left = False
    threat_right = False
    for proj in env.alien_projectiles:
        if (player_y - proj.y) < 50 and abs(proj.x - player_x) < 20:
            if proj.x < player_x:
                threat_left = True
            else:
                threat_right = True
    
    # Dodge projectiles if threatened
    if threat_left and threat_right:
        # If threatened from both sides, move toward clearer side
        left_count = sum(1 for p in env.alien_projectiles if p.x < player_x)
        right_count = sum(1 for p in env.alien_projectiles if p.x >= player_x)
        movement = 3 if left_count < right_count else 4
    elif threat_left:
        movement = 4  # Move right
    elif threat_right:
        movement = 3  # Move left
    else:
        # Target nearest alien if no immediate threats
        if env.aliens:
            nearest_alien = min(env.aliens, key=lambda a: abs(a['rect'].centerx - player_x))
            if nearest_alien['rect'].centerx < player_x:
                movement = 3
            elif nearest_alien['rect'].centerx > player_x:
                movement = 4
            else:
                movement = 0
        else:
            movement = 0
    
    # Fire only when cooldown allows and aliens exist
    fire = 1 if env.player_fire_cooldown == 0 and env.aliens else 0
    
    return [movement, fire, 0]