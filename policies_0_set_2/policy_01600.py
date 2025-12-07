def policy(env):
    """
    Strategy: Maximize reward by aligning paddle with predicted ball position when descending, 
    and centering paddle when ascending. Avoids penalties by moving towards ball when descending.
    Secondary actions (a1, a2) are unused in this environment and set to 0.
    """
    if env.game_over:
        return [0, 0, 0]
    
    paddle_center = env.paddle_rect.centerx
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    
    if ball_vy > 0:  # Ball descending
        time_to_paddle = (env.paddle_rect.top - ball_y) / ball_vy
        predicted_x = ball_x + ball_vx * time_to_paddle
        # Clamp prediction to screen bounds
        paddle_half = env.PADDLE_WIDTH / 2
        target_x = max(paddle_half, min(env.SCREEN_WIDTH - paddle_half, predicted_x))
        
        if abs(paddle_center - target_x) < 5:
            move = 0
        elif paddle_center < target_x:
            move = 4
        else:
            move = 3
    else:  # Ball ascending or horizontal
        center = env.SCREEN_WIDTH / 2
        if abs(paddle_center - center) < 5:
            move = 0
        elif paddle_center < center:
            move = 4
        else:
            move = 3
            
    return [move, 0, 0]