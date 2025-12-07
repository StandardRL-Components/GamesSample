def policy(env):
    # Strategy: Launch ball immediately, then predict ball's future x-position based on velocity and bounces.
    # Move paddle to intercept, maximizing block breaks while minimizing ball loss.
    if not env.ball_launched:
        return [0, 1, 0]  # Launch ball
    else:
        ball_x, ball_y = env.ball_pos
        ball_vx, ball_vy = env.ball_vel
        paddle_center = env.paddle.x + env.PADDLE_WIDTH / 2
        paddle_y = env.HEIGHT - env.PADDLE_HEIGHT - 10
        
        if ball_vy > 0 and ball_y < paddle_y:  # Ball moving down and above paddle
            time_to_paddle = (paddle_y - ball_y) / ball_vy
            predicted_x = ball_x + ball_vx * time_to_paddle
            low_x = env.BALL_RADIUS
            high_x = env.WIDTH - env.BALL_RADIUS
            range_x = high_x - low_x
            period = 2 * range_x
            x_normalized = (predicted_x - low_x) % period
            if x_normalized < 0:
                x_normalized += period
            if x_normalized > range_x:
                x_normalized = period - x_normalized
            predicted_x = x_normalized + low_x
        else:
            predicted_x = ball_x
            
        if paddle_center < predicted_x - 5:
            return [4, 0, 0]  # Move right
        elif paddle_center > predicted_x + 5:
            return [3, 0, 0]  # Move left
        else:
            return [0, 0, 0]  # No movement