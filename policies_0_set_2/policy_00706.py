def policy(env):
    # Strategy: Use internal state tracking for precise paddle alignment with ball trajectory.
    # Prioritize launching ball when held, then intercept based on predicted bounce position.
    # Leverage exact env attributes (read-only) for reliable state information over pixel parsing.
    
    # If ball is held, launch it and center paddle for better coverage
    if env.ball_held:
        paddle_center = env.paddle.centerx
        screen_center = env.WIDTH / 2
        if paddle_center < screen_center - 5:
            return [4, 1, 0]  # Right + launch
        elif paddle_center > screen_center + 5:
            return [3, 1, 0]  # Left + launch
        else:
            return [0, 1, 0]   # Launch only

    # Predict ball's landing position using physics simulation
    L = env.WIDTH - 2 * env.WALL_THICKNESS
    if env.ball_vel.y > 0:  # Ball moving downward
        time_to_impact = (env.paddle.top - env.ball_pos.y) / env.ball_vel.y
        future_x = env.ball_pos.x + env.ball_vel.x * time_to_impact
        
        # Account for wall bounces using reflection principle
        x_rel = (future_x - env.WALL_THICKNESS) % (2 * L)
        if x_rel > L:
            x_rel = 2 * L - x_rel
        target_x = x_rel + env.WALL_THICKNESS
    else:
        # Ball moving upward - center paddle while waiting
        target_x = env.WIDTH / 2

    # Align paddle with predicted impact point
    paddle_center = env.paddle.centerx
    if paddle_center < target_x - 5:
        return [4, 0, 0]  # Move right
    elif paddle_center > target_x + 5:
        return [3, 0, 0]  # Move left
    else:
        return [0, 0, 0]   # Maintain position