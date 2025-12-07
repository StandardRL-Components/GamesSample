def policy(env):
    """Policy for Breakout: Predict ball's future x-position when moving down and move paddle accordingly. Otherwise, do nothing."""
    if env.game_over:
        return [0, 0, 0]
    
    if env.ball_vel[1] > 0 and env.ball_pos[1] < env.paddle.y:
        t = (env.paddle.y - env.ball_pos[1]) / env.ball_vel[1]
        future_x = env.ball_pos[0] + env.ball_vel[0] * t
        
        # Account for wall bounces
        time_remaining = t
        current_x = env.ball_pos[0]
        current_vx = env.ball_vel[0]
        while time_remaining > 0 and abs(current_vx) > 1e-5:
            if current_vx > 0:
                time_to_wall = (env.SCREEN_WIDTH - env.BALL_RADIUS - current_x) / current_vx
            else:
                time_to_wall = (current_x - env.BALL_RADIUS) / -current_vx
            
            if time_to_wall > time_remaining:
                current_x += current_vx * time_remaining
                break
            else:
                current_x += current_vx * time_to_wall
                time_remaining -= time_to_wall
                current_vx = -current_vx
        
        future_x = current_x
        future_x = max(env.PADDLE_WIDTH/2, min(future_x, env.SCREEN_WIDTH - env.PADDLE_WIDTH/2))
        paddle_center = env.paddle.x + env.PADDLE_WIDTH/2
        
        if abs(future_x - paddle_center) < env.PADDLE_SPEED:
            movement = 0
        else:
            movement = 3 if future_x < paddle_center else 4
    else:
        movement = 0
    
    return [movement, 0, 0]