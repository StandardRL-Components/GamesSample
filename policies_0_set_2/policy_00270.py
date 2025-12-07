def policy(env):
    """
    Navigates toward the goal (right side) while avoiding obstacles. Prioritizes rightward movement
    unless blocked, then uses vertical movements to avoid obstacles. Uses hyper-jump when path is
    clear to cover distance quickly, and activates shield only when collisions are imminent to
    conserve time. Always prefers rightward progress when safe.
    """
    player_pos = env.player_pos
    goal_center = pygame.Vector2(env.goal_rect.center)
    
    # Calculate basic rightward movement preference
    right_vec = pygame.Vector2(1, 0)
    up_vec = pygame.Vector2(0, -1)
    down_vec = pygame.Vector2(0, 1)
    
    # Check obstacle density in each direction
    right_obstacles = sum(1 for o in env.obstacles 
                         if (o['pos'] - player_pos).dot(right_vec) > 0 
                         and player_pos.distance_to(o['pos']) < 60)
    up_obstacles = sum(1 for o in env.obstacles 
                      if (o['pos'] - player_pos).dot(up_vec) > 0 
                      and player_pos.distance_to(o['pos']) < 40)
    down_obstacles = sum(1 for o in env.obstacles 
                        if (o['pos'] - player_pos).dot(down_vec) > 0 
                        and player_pos.distance_to(o['pos']) < 40)
    
    # Determine movement direction
    if right_obstacles == 0:
        movement = 4  # Right
    elif up_obstacles <= down_obstacles:
        movement = 1  # Up
    else:
        movement = 2  # Down
    
    # Use hyper-jump if available and path is clear
    hyper_jump = 0
    if (env.hop_cooldown == 0 and 
        env.time_remaining_ticks > env.HYPER_JUMP_COST_TICKS and
        right_obstacles == 0):
        hyper_jump = 1
    
    # Activate shield only when necessary to avoid collision
    shield = 0
    if (not env.shield_active and 
        env.time_remaining_ticks > env.SHIELD_COST_TICKS and
        any(player_pos.distance_to(o['pos']) < 25 for o in env.obstacles)):
        shield = 1
    
    return [movement, hyper_jump, shield]