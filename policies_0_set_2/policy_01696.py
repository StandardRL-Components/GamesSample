def policy(env):
    # Strategy: Track ball's projected landing point when moving down, otherwise center paddle.
    # Launch immediately if ball is attached. Prioritize preventing ball loss by predicting intercept point.
    if env.ball_attached:
        return [0, 1, 0]  # Launch ball immediately when attached
    
    if env.ball_vel.y > 0:  # Ball moving downward
        # Predict where ball will hit paddle level
        time_to_paddle = (env.PADDLE_Y - env.ball_pos.y) / env.ball_vel.y
        target_x = env.ball_pos.x + env.ball_vel.x * time_to_paddle
        
        # Account for wall bounces
        while target_x < 0 or target_x > env.WIDTH:
            target_x = 2 * (env.WIDTH if target_x > env.WIDTH else 0) - target_x
        
        # Move paddle toward predicted intercept point
        paddle_center = env.paddle.centerx
        if paddle_center < target_x - 5:
            return [4, 0, 0]  # Right
        elif paddle_center > target_x + 5:
            return [3, 0, 0]  # Left
        else:
            return [0, 0, 0]  # No movement
    
    # Ball moving upward - center paddle for defense
    paddle_center = env.paddle.centerx
    if paddle_center < env.WIDTH / 2 - 5:
        return [4, 0, 0]  # Right
    elif paddle_center > env.WIDTH / 2 + 5:
        return [3, 0, 0]  # Left
    else:
        return [0, 0, 0]  # No movement