def policy(env):
    # Strategy: Track the ball's predicted landing point to position the paddle for optimal rebounds.
    # Prioritize intercepting the ball to break bricks and avoid missing, maximizing score and survival rewards.
    paddle_center = env.paddle.centerx
    ball_center = env.ball.centerx
    ball_velocity_x = env.ball_vel[0]
    
    # Predict ball's x-position when it reaches paddle height if moving downward
    if env.ball_vel[1] > 0:  # Ball moving downward
        time_to_paddle = (env.paddle.top - env.ball.centery) / env.ball_vel[1]
        predicted_x = ball_center + ball_velocity_x * time_to_paddle
        
        # Account for wall bounces by reflecting predicted position within game bounds
        game_width = env.game_area.right - env.game_area.left
        normalized_x = (predicted_x - env.game_area.left) % (2 * game_width)
        if normalized_x > game_width:
            normalized_x = 2 * game_width - normalized_x
        target_x = env.game_area.left + normalized_x
    else:
        # Ball moving upward, center paddle defensively
        target_x = env.game_area.left + (env.game_area.right - env.game_area.left) / 2
    
    # Move toward target position with deadzone to prevent oscillation
    if target_x < paddle_center - 10:
        move_action = 3  # Left
    elif target_x > paddle_center + 10:
        move_action = 4  # Right
    else:
        move_action = 0  # None
    
    return [move_action, 0, 0]  # a1 and a2 unused in this environment