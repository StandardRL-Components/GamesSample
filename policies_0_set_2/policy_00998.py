def policy(env):
    # Prioritize survival by avoiding projectiles, then align with closest alien to maximize shooting efficiency
    player_y = env.player.pos.y
    player_height = env.player.h
    player_center = player_y + player_height / 2
    
    # Check for immediate projectile threats
    danger_above = False
    danger_below = False
    for proj in env.alien_projectiles:
        if proj.pos.x < 100:  # Projectiles within dangerous range
            if proj.pos.y < player_center:
                danger_above = True
            else:
                danger_below = True
    
    # Avoid projectiles with priority
    if danger_above and not danger_below:
        movement = 2  # Move down
    elif danger_below and not danger_above:
        movement = 1  # Move up
    elif danger_above and danger_below:
        movement = 1 if player_y < env.HEIGHT / 2 else 2  # Move toward center
    else:
        # Align with closest alien when no immediate danger
        closest_alien = None
        min_dist = float('inf')
        for alien in env.aliens:
            dist = abs(alien.pos.y + alien.h/2 - player_center)
            if dist < min_dist:
                min_dist = dist
                closest_alien = alien
        
        if closest_alien:
            alien_center = closest_alien.pos.y + closest_alien.h/2
            if alien_center < player_center - 5:
                movement = 1  # Move up
            elif alien_center > player_center + 5:
                movement = 2  # Move down
            else:
                movement = 0  # Maintain position
        else:
            movement = 0  # No aliens visible
    
    # Always fire when possible
    fire = 1 if env.player.can_fire() else 0
    
    return [movement, fire, 0]