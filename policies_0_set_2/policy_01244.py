def policy(env):
    # Strategy: Track the ball's predicted x-position when it reaches the paddle height,
    # adjusting for wall bounces. Move the paddle to intercept the ball to maximize breaks and rewards.
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    paddle_x = env.paddle_rect.x
    paddle_width = env.paddle_rect.width
    wall_thickness = env.WALL_THICKNESS
    screen_width = env.SCREEN_WIDTH

    if ball_vy > 0:  # Ball moving downward
        # Predict x-coordinate when ball reaches paddle height
        time_to_paddle = (env.paddle_rect.top - ball_y) / ball_vy
        predicted_x = ball_x + ball_vx * time_to_paddle
        
        # Account for wall bounces
        arena_width = screen_width - 2 * wall_thickness
        normalized_x = (predicted_x - wall_thickness) % (2 * arena_width)
        if normalized_x > arena_width:
            normalized_x = 2 * arena_width - normalized_x
        predicted_x = wall_thickness + normalized_x
        
        target_x = predicted_x - paddle_width / 2
    else:  # Ball moving upward, center paddle
        target_x = (screen_width - paddle_width) / 2

    # Determine movement direction
    current_center = paddle_x + paddle_width / 2
    if abs(current_center - target_x) < 5:  # Deadzone to prevent oscillation
        move_action = 0
    elif target_x < current_center:
        move_action = 3
    else:
        move_action = 4

    return [move_action, 0, 0]  # a1 and a2 unused in this environment