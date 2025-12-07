def policy(env):
    # Strategy: Prioritize survival by dodging projectiles while maintaining constant fire.
    # Use shield only when projectiles are imminent and unavoidable. Move horizontally to align
    # with alien clusters for efficient shooting while avoiding incoming fire.
    
    # Read current game state
    player = env.player
    enemy_projectiles = env.enemy_projectiles
    aliens = env.aliens
    
    # Initialize actions
    movement = 0  # No movement
    fire = 1 if player['fire_cooldown'] == 0 else 0  # Fire if available
    shield = 0
    
    # Calculate player position
    player_x = player['rect'].centerx
    player_y = player['rect'].centery
    
    # Check for imminent projectile threats
    for proj in enemy_projectiles:
        proj_x, proj_y = proj['rect'].centerx, proj['rect'].centery
        # If projectile is close and heading toward player
        if (abs(proj_x - player_x) < 20 and 
            proj_y > player_y - 50 and 
            proj_y < player_y + 10):
            # Activate shield if available and not active
            if (player['shield_cooldown'] == 0 and 
                player['shield_active_timer'] == 0):
                shield = 1
            # Dodge horizontally if shield unavailable
            elif proj_x < player_x:
                movement = 4  # Move right
            else:
                movement = 3  # Move left
            break
    
    # If no immediate threat, target nearest alien cluster
    if movement == 0 and aliens:
        # Find average x-position of aliens
        avg_alien_x = sum(a['rect'].centerx for a in aliens) / len(aliens)
        if avg_alien_x < player_x - 10:
            movement = 3  # Move left
        elif avg_alien_x > player_x + 10:
            movement = 4  # Move right
    
    return [movement, fire, shield]