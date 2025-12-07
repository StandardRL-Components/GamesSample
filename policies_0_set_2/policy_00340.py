def policy(env):
    """
    Prioritizes collecting stars by jumping towards nearest star with appropriate power.
    Uses high-power jumps for distant stars, low-power for nearby ones. Fast-falls when
    above platforms to minimize air time and fuel waste. Targets immediate rewards (stars)
    while conserving fuel for essential jumps.
    """
    if env.game_over:
        return [0, 0, 0]
    
    movement = 0
    shift_held = 0
    
    if env.on_ground and env.fuel > 0 and env.stars:
        # Find nearest star
        min_dist = float('inf')
        target_star = None
        for star in env.stars:
            dx = star[0] - env.player_pos.x
            dy = star[1] - env.player_pos.y
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                target_star = star
        
        dx = target_star[0] - env.player_pos.x
        dy = target_star[1] - env.player_pos.y
        
        # Choose jump direction
        if abs(dx) < 20:
            movement = 1  # Jump up for nearly vertical targets
        else:
            movement = 3 if dx < 0 else 4  # Jump left/right for horizontal movement
        
        # Use high power for distant targets
        if abs(dx) > 50 or abs(dy) > 50:
            shift_held = 1
    
    elif not env.on_ground and env.player_vel.y > 0:
        # Fast fall when above platforms
        player_rect = env._get_player_rect()
        check_height = 100
        check_rect = pygame.Rect(player_rect.x, player_rect.bottom, player_rect.width, check_height)
        for plat in env.platforms:
            if check_rect.colliderect(plat):
                movement = 2
                break
    
    return [movement, 0, shift_held]