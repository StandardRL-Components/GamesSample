def policy(env):
    # Strategy: Maximize score by directing gravity upward to accelerate ball towards opponent, 
    # and move paddle to intercept ball's predicted landing point based on current trajectory.
    gravity_action = 1  # Always set gravity to upward (a0=1) to encourage ball movement towards opponent
    ball_vel_y = env.ball_vel[1]
    paddle_center_x = env.player_paddle.centerx
    screen_center_x = env.WIDTH / 2
    
    # Predict ball's future x-position when it reaches paddle height if moving downward
    if ball_vel_y > 0 and env.ball.centery < env.player_paddle.top:
        time_to_reach = (env.player_paddle.top - env.ball.centery) / ball_vel_y
        future_x = env.ball.centerx + env.ball_vel[0] * time_to_reach
        future_x = max(env.player_paddle.width/2, min(env.WIDTH - env.player_paddle.width/2, future_x))
    else:
        future_x = screen_center_x  # Default to center if ball moving up or unreachable
    
    # Move paddle toward predicted position with 5-pixel tolerance to avoid oscillation
    tolerance = 5
    if paddle_center_x < future_x - tolerance:
        move_left, move_right = 0, 1
    elif paddle_center_x > future_x + tolerance:
        move_left, move_right = 1, 0
    else:
        move_left, move_right = 0, 0
    
    return [gravity_action, move_left, move_right]