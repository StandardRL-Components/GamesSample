def policy(env):
    # Strategy: Prioritize survival by avoiding projectiles, then destroy enemies by aligning and firing.
    # Always fire when possible to maximize damage. Move horizontally to dodge threats or align with enemies.
    
    # Check if firing is available
    fire_action = 1 if env.player_fire_cooldown == 0 else 0
    
    # Check for immediate projectile threats
    dangerous_projectiles = []
    for proj in env.enemy_projectiles:
        if (proj['rect'].bottom < env.player_rect.top and 
            abs(proj['rect'].centerx - env.player_rect.centerx) < 50):
            dangerous_projectiles.append(proj)
    
    if dangerous_projectiles:
        # Dodge the nearest threatening projectile
        nearest_proj = min(dangerous_projectiles, 
                          key=lambda p: abs(p['rect'].centerx - env.player_rect.centerx))
        if nearest_proj['rect'].centerx > env.player_rect.centerx:
            movement = 3  # Move left
        else:
            movement = 4  # Move right
    else:
        # Find closest enemy to target
        if env.enemies:
            closest_enemy = min(env.enemies, 
                               key=lambda e: abs(e['rect'].centerx - env.player_rect.centerx))
            dx = closest_enemy['rect'].centerx - env.player_rect.centerx
            if abs(dx) > 20:
                movement = 3 if dx < 0 else 4
            else:
                movement = 0  # Stay aligned
        else:
            movement = 0
    
    return [movement, fire_action, 0]