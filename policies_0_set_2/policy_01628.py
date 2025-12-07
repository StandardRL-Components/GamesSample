def policy(env):
    # Prioritize accelerating towards next checkpoint while avoiding obstacles. Use boost on straightaways and brake when misaligned or near obstacles.
    checkpoint = env.track_nodes[env.next_checkpoint_index]
    dx = checkpoint.x - env.player_pos.x
    dy = checkpoint.y - env.player_pos.y
    
    # Choose movement direction toward checkpoint
    movement_scores = [0, -dy, dy, -dx, dx]
    best_movement = 0
    best_score = movement_scores[0]
    for i in range(1, 5):
        if movement_scores[i] > best_score:
            best_score = movement_scores[i]
            best_movement = i
    
    # Check obstacle proximity and alignment
    lookahead_time = 1.0
    future_x = env.player_pos.x + env.player_vel.x * lookahead_time
    future_y = env.player_pos.y + env.player_vel.y * lookahead_time
    obstacle_near = False
    for obs in env.obstacles:
        obs_dx = future_x - obs.x
        obs_dy = future_y - obs.y
        dist_sq = obs_dx**2 + obs_dy**2
        if dist_sq < (env.OBSTACLE_RADIUS + 20)**2:
            obstacle_near = True
            break
    
    current_speed_sq = env.player_vel.x**2 + env.player_vel.y**2
    aligned = False
    if current_speed_sq > 0 and (dx != 0 or dy != 0):
        dot_product = env.player_vel.x * dx + env.player_vel.y * dy
        desired_sq = dx**2 + dy**2
        aligned = dot_product**2 > 0.81 * current_speed_sq * desired_sq
    
    # Apply boost when aligned and clear path, brake when misaligned or near obstacles
    boost = 1 if aligned and not obstacle_near and current_speed_sq < (env.MAX_SPEED * 0.9)**2 else 0
    brake = 1 if obstacle_near or (current_speed_sq > (env.MAX_SPEED * 0.5)**2 and not aligned) else 0
    
    return [best_movement, boost, brake]