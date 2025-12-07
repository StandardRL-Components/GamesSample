def policy(env):
    # Strategy: Predict ball trajectory to align paddle, using special shots when misaligned near right edge to avoid misses and maximize scoring.
    ball_x, ball_y = env.ball_pos
    ball_vel_x, ball_vel_y = env.ball_vel
    paddle_center = env.paddle.y + env.PADDLE_HEIGHT / 2
    
    # Predict ball's future y-position at paddle's x-coordinate
    if ball_vel_x > 0:  # Ball moving toward paddle
        time_to_paddle = (env.paddle.x - ball_x) / ball_vel_x
        future_y = ball_y + ball_vel_y * time_to_paddle
        # Account for wall bounces by clamping predicted y
        future_y = max(0, min(env.GRID_HEIGHT - 1, future_y))
        target_y = future_y
    else:
        target_y = env.GRID_HEIGHT / 2  # Default to center when ball moving away
    
    # Move paddle toward predicted position
    if target_y < paddle_center - 0.5:
        movement = 1  # Up
    elif target_y > paddle_center + 0.5:
        movement = 2  # Down
    else:
        movement = 0  # No movement
    
    # Use special shot if available and ball is near right edge with misalignment
    fire_special = False
    if (env.special_shots > 0 and env.special_shot_effect is None and 
        ball_x > env.GRID_WIDTH - 2 and abs(paddle_center - ball_y) > 1.0):
        fire_special = True
    
    return [movement, 1 if fire_special else 0, 0]