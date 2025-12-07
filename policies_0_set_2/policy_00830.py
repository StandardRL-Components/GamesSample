def policy(env):
    # Strategy: Track the ball's predicted landing point when moving down, otherwise center the paddle.
    # Maximizes reward by preventing ball loss and breaking bricks efficiently.
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    paddle_x = env.paddle.x + env.PADDLE_WIDTH / 2
    
    if ball_vy > 0:  # Ball moving downward
        # Predict x-coordinate when ball reaches paddle height
        time_to_paddle = (env.paddle.top - ball_y) / (ball_vy * env.current_ball_speed)
        predicted_x = ball_x + ball_vx * env.current_ball_speed * time_to_paddle
        
        # Account for wall bounces
        while predicted_x < 0 or predicted_x > env.SCREEN_WIDTH:
            if predicted_x < 0:
                predicted_x = -predicted_x
            else:
                predicted_x = 2 * env.SCREEN_WIDTH - predicted_x
        
        target_x = predicted_x - env.PADDLE_WIDTH / 2
    else:  # Ball moving upward
        target_x = env.SCREEN_WIDTH / 2 - env.PADDLE_WIDTH / 2  # Center paddle
    
    # Choose movement direction
    if paddle_x < target_x - 5:
        move_action = 4  # Right
    elif paddle_x > target_x + 5:
        move_action = 3  # Left
    else:
        move_action = 0  # None
    
    return [move_action, 0, 0]  # a1 and a2 unused in this game