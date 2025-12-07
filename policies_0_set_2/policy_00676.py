def policy(env):
    # Strategy: Track ball's vertical position when moving towards paddle to maximize hits and minimize misses.
    # Align paddle center with predicted ball Y position, accounting for bounces, to intercept efficiently.
    if env.game_over:
        return [0, 0, 0]
    
    paddle_center_y = env.paddle_rect.centery
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    
    # Predict ball's Y position when it reaches paddle using current trajectory and wall bounces
    if ball_vx > 0:  # Ball moving towards paddle
        time_to_reach = (env.paddle_rect.left - ball_x) / ball_vx
        predicted_y = ball_y + ball_vy * time_to_reach
        
        # Account for vertical wall bounces
        while predicted_y < env.BALL_RADIUS or predicted_y > env.SCREEN_HEIGHT - env.BALL_RADIUS:
            if predicted_y < env.BALL_RADIUS:
                predicted_y = 2 * env.BALL_RADIUS - predicted_y
                ball_vy = -ball_vy
            else:
                predicted_y = 2 * (env.SCREEN_HEIGHT - env.BALL_RADIUS) - predicted_y
                ball_vy = -ball_vy
        
        target_y = predicted_y
    else:
        # Ball moving away - center paddle for next approach
        target_y = env.SCREEN_HEIGHT / 2
    
    # Move paddle towards target position
    if abs(paddle_center_y - target_y) < 5:
        move_action = 0  # No movement if close enough
    elif paddle_center_y < target_y:
        move_action = 2  # Move down
    else:
        move_action = 1  # Move up
    
    return [move_action, 0, 0]