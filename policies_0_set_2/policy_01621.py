def policy(env):
    # Strategy: Track ball's x-position when moving downward to align paddle. Launch immediately.
    # Prioritize keeping ball in play by predicting landing position based on current trajectory.
    if not env.ball_launched:
        return [0, 1, 0]  # No movement, launch ball
    
    # Predict ball's x-position when it reaches paddle height
    if env.ball_vel[1] > 0:  # Ball moving downward
        time_to_paddle = (env.paddle.top - env.ball.centery) / env.ball_vel[1]
        predicted_x = env.ball.centerx + env.ball_vel[0] * time_to_paddle
        
        # Account for wall bounces
        while predicted_x < 0 or predicted_x > env.WIDTH:
            if predicted_x < 0:
                predicted_x = -predicted_x
            else:
                predicted_x = 2 * env.WIDTH - predicted_x
        
        # Move paddle to predicted position
        paddle_center = env.paddle.x + env.paddle.width / 2
        if abs(paddle_center - predicted_x) > 10:  # Deadzone to prevent oscillation
            if paddle_center < predicted_x:
                return [4, 0, 0]  # Move right
            else:
                return [3, 0, 0]  # Move left
    
    return [0, 0, 0]  # No movement needed