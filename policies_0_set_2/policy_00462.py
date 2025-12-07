def policy(env):
    # Strategy: Track ball's x-position when moving downward to intercept, otherwise center paddle. 
    # This minimizes missed balls (costing lives) and maximizes block breaks (+1.1 reward) while reducing movement penalties.
    ball_y_vel = env.ball_vel.y
    paddle_center = env.paddle.centerx
    
    if ball_y_vel > 0:  # Ball moving downward - track its x-position
        target_x = env.ball_pos.x
    else:  # Ball moving upward - center paddle for better coverage
        target_x = env.SCREEN_WIDTH / 2
    
    diff = target_x - paddle_center
    if abs(diff) < 10:  # Deadzone to minimize movement penalties
        a0 = 0
    else:
        a0 = 3 if diff < 0 else 4
    
    # Prevent movement if at boundary to avoid unnecessary penalties
    if (a0 == 3 and env.paddle.x <= env.WALL_THICKNESS) or (a0 == 4 and env.paddle.x >= env.SCREEN_WIDTH - env.WALL_THICKNESS - env.PADDLE_WIDTH):
        a0 = 0
    
    return [a0, 0, 0]