def policy(env):
    # Strategy: Launch ball immediately, then track ball's x-position when moving downward.
    # Maximizes score by keeping ball in play with simple reactive tracking.
    try:
        if not env.ball_launched:
            return [0, 1, 0]  # Launch ball without moving
        
        ball_x, ball_y = env.ball.centerx, env.ball.centery
        paddle_x = env.paddle.centerx
        
        # Only track ball when moving downward below mid-screen
        if env.ball_vel[1] > 0 and ball_y > env.SCREEN_HEIGHT / 2:
            if paddle_x < ball_x - 10:
                return [4, 0, 0]  # Move right
            elif paddle_x > ball_x + 10:
                return [3, 0, 0]  # Move left
        
        return [0, 0, 0]  # Default no-op
    
    except:
        return [0, 0, 0]  # Fallback no-op