def policy(env):
    """
    Breakout strategy: Track ball's predicted landing position to align paddle.
    Maximizes reward by preventing ball loss and breaking blocks efficiently.
    Uses linear extrapolation with wall bounce reflection for accurate interception.
    """
    # If game over, no movement needed
    if env.game_over:
        return [0, 0, 0]
    
    ball_x, ball_y = env.ball_pos
    vel_x, vel_y = env.ball_vel
    paddle_center = env.paddle.centerx
    
    # Predict ball's x-position at paddle height
    if vel_y > 0:  # Ball moving downward
        time_to_paddle = (env.paddle.y - ball_y) / vel_y
        future_x = ball_x + vel_x * time_to_paddle
        
        # Account for wall bounces using reflection
        effective_width = env.SCREEN_WIDTH - 2 * env.BALL_RADIUS
        rel_x = (future_x - env.BALL_RADIUS) % (2 * effective_width)
        if rel_x > effective_width:
            rel_x = 2 * effective_width - rel_x
        future_x = rel_x + env.BALL_RADIUS
    else:
        # Ball moving upward, center paddle
        future_x = env.SCREEN_WIDTH / 2
    
    # Move paddle toward predicted position
    if future_x < paddle_center - 5:
        return [3, 0, 0]  # Left
    elif future_x > paddle_center + 5:
        return [4, 0, 0]  # Right
    else:
        return [0, 0, 0]  # No movement