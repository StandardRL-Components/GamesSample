def policy(env):
    # Strategy: Prioritize survival by dodging projectiles while targeting aliens. 
    # Move horizontally to align with aliens and fire continuously when aligned. 
    # Avoid unnecessary movement to minimize risk from projectiles.
    player_x = env.player_rect.centerx
    player_y = env.player_rect.centery
    
    # Check for immediate projectile threats
    dangerous_projectiles = []
    for proj in env.projectiles:
        if proj['type'] == 'alien' and proj['rect'].centery > player_y - 50:
            dangerous_projectiles.append(proj)
    
    # Dodge projectiles if any are close
    if dangerous_projectiles:
        closest_proj = min(dangerous_projectiles, key=lambda p: abs(p['rect'].centerx - player_x))
        if closest_proj['rect'].centerx < player_x:
            movement = 4  # Move right if projectile is left
        else:
            movement = 3  # Move left if projectile is right
    else:
        # Target nearest alien when no immediate threats
        if env.aliens:
            nearest_alien = min(env.aliens, key=lambda a: abs(a['rect'].centerx - player_x))
            if nearest_alien['rect'].centerx < player_x - 10:
                movement = 3  # Move left
            elif nearest_alien['rect'].centerx > player_x + 10:
                movement = 4  # Move right
            else:
                movement = 0  # Stay aligned
        else:
            movement = 0  # No movement if no aliens
    
    # Always fire when possible (a1=1) and never use secondary action (a2=0)
    fire = 1 if env.player_fire_timer == 0 else 0
    return [movement, fire, 0]