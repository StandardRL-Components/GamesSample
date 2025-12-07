def policy(env):
    # Strategy: Track ball x-position and velocity to predict landing point. Move paddle to intercept 
    # when ball is moving downward, otherwise center paddle. Launch ball immediately when on paddle.
    paddle = env.paddle
    ball_pos = env.ball_pos
    ball_vel = env.ball_vel
    
    if env.ball_on_paddle:
        return [0, 1, 0]  # Launch ball immediately
    
    if ball_vel[1] > 0:  # Ball moving downward
        # Predict x-coordinate when ball reaches paddle height
        time_to_paddle = (paddle.top - ball_pos[1]) / ball_vel[1]
        predicted_x = ball_pos[0] + ball_vel[0] * time_to_paddle
        
        # Account for wall bounces
        while predicted_x < 0 or predicted_x > env.WIDTH:
            if predicted_x < 0:
                predicted_x = -predicted_x
            else:
                predicted_x = 2 * env.WIDTH - predicted_x
        
        # Move toward predicted position
        if abs(paddle.centerx - predicted_x) < env.PADDLE_SPEED:
            return [0, 0, 0]
        return [4 if paddle.centerx < predicted_x else 3, 0, 0]
    else:
        # Ball moving upward - center paddle
        center = env.WIDTH // 2
        if abs(paddle.centerx - center) < env.PADDLE_SPEED:
            return [0, 0, 0]
        return [4 if paddle.centerx < center else 3, 0, 0]