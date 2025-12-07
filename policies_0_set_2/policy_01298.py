def policy(env):
    # Strategy: Prioritize dodging incoming projectiles to preserve lives, then align with nearest alien to shoot.
    # Fire when cooldown allows to maximize alien destruction. Movement avoids borders and targets closest threat.
    player_pos = env.player_pos
    aliens = env.aliens
    projectiles = env.alien_projectiles
    
    # Initialize actions: no movement, fire if possible, no secondary action
    a0 = 0
    a1 = 1 if (env.fire_cooldown == 0 and not env.prev_space_held) else 0
    a2 = 0
    
    # Dodge nearby projectiles first (within 20 pixels horizontally and above player)
    for proj in projectiles:
        if abs(proj[0] - player_pos[0]) < 20 and proj[1] < player_pos[1]:
            if proj[0] > player_pos[0]:
                a0 = 3  # move left
            else:
                a0 = 4  # move right
            return [a0, a1, a2]
    
    # Find closest alien above player to target
    min_dist = float('inf')
    target = None
    for alien in aliens:
        if alien['pos'][1] < player_pos[1]:  # Only consider aliens above player
            dist = abs(alien['pos'][0] - player_pos[0])
            if dist < min_dist:
                min_dist = dist
                target = alien
    
    # Move toward closest alien if found
    if target is not None:
        if target['pos'][0] < player_pos[0] and player_pos[0] > env.BORDER + env.PLAYER_SPEED:
            a0 = 3  # left
        elif target['pos'][0] > player_pos[0] and player_pos[0] < env.WIDTH - env.BORDER - env.PLAYER_SPEED:
            a0 = 4  # right
    
    return [a0, a1, a2]