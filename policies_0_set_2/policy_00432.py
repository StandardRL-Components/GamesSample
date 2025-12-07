def policy(env):
    # Strategy: Prioritize shooting aliens for immediate rewards, dodge incoming shots by moving horizontally,
    # and use shield only when necessary to avoid shots that can't be dodged. Move towards alien clusters to maximize hit rate.
    movement = 0  # Default: no movement
    fire = 0
    shield = 0
    
    # Get current game state from env attributes (read-only)
    player_pos = env.player_pos
    aliens = env.aliens
    alien_shots = env.alien_shots
    shoot_cooldown = env.player_shoot_timer
    shield_cooldown = env.shield_cooldown_timer
    shield_active = env.shield_active_timer
    
    # Always shoot if cooldown is ready and aliens exist
    if shoot_cooldown == 0 and aliens:
        fire = 1
    
    # Dodge incoming shots by moving away from nearest threat
    closest_shot = None
    min_dist = float('inf')
    for shot in alien_shots:
        dist = (shot[0]-player_pos[0])**2 + (shot[1]-player_pos[1])**2
        if dist < min_dist:
            min_dist = dist
            closest_shot = shot
    
    if closest_shot:
        # Move horizontally away from shot
        if closest_shot[0] < player_pos[0]:
            movement = 4  # Right
        else:
            movement = 3  # Left
        
        # Use shield if shot is too close and shield available
        if min_dist < 900 and shield_cooldown == 0 and shield_active == 0:
            shield = 1
    else:
        # Move toward alien cluster center when no immediate threats
        if aliens:
            avg_x = sum(a['pos'][0] for a in aliens) / len(aliens)
            if avg_x < player_pos[0] - 10:
                movement = 3  # Left
            elif avg_x > player_pos[0] + 10:
                movement = 4  # Right
    
    return [movement, fire, shield]