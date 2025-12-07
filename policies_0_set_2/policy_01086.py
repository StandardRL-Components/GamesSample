def policy(env):
    # Strategy: Track ball trajectory to position paddle for optimal rebounds, breaking blocks efficiently.
    # Prioritizes catching the ball to prevent life loss while aiming to clear blocks for maximum reward.
    ball_x, ball_y = env.ball_pos
    vel_x, vel_y = env.ball_vel
    paddle_center = env.paddle.centerx
    paddle_y = env.paddle.y
    
    if env.ball_attached:
        # Launch ball immediately with neutral angle to start breaking blocks
        return [0, 1, 0]
    
    # Predict ball's future x-position when it reaches paddle height
    if vel_y > 0:  # Ball moving downward
        time_to_paddle = (paddle_y - ball_y) / vel_y
        future_x = ball_x + vel_x * time_to_paddle
        
        # Account for wall bounces
        screen_width = env.SCREEN_WIDTH
        ball_radius = env.BALL_RADIUS
        while future_x < ball_radius or future_x > screen_width - ball_radius:
            if future_x < ball_radius:
                future_x = 2 * ball_radius - future_x
            else:
                future_x = 2 * (screen_width - ball_radius) - future_x
            vel_x *= -1
        
        # Move paddle toward predicted impact point
        if paddle_center < future_x - 5:
            return [4, 0, 0]
        elif paddle_center > future_x + 5:
            return [3, 0, 0]
    
    # Default: minimal movement to avoid oscillation
    return [0, 0, 0]