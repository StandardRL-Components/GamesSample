def policy(env):
    # Strategy: Predict ball's landing point when moving down to hit and break bricks for reward; move to center when ball is up. Avoid oscillation with dead zone.
    ball_x, ball_y = env.ball_pos
    vx, vy = env.ball_vel
    paddle_x, paddle_y = env.paddle.x, env.paddle.y
    paddle_width, ball_radius = env.PADDLE_WIDTH, env.BALL_RADIUS
    width, height = env.WIDTH, env.HEIGHT
    left_bound, right_bound = ball_radius, width - ball_radius
    
    if vy > 0:  # Ball moving down
        target_y = paddle_y - ball_radius
        t = (target_y - ball_y) / vy
        predicted_x = ball_x + vx * t
        predicted_x = max(left_bound, min(predicted_x, right_bound))
    else:  # Ball moving up or stationary
        predicted_x = width / 2  # Move to center
    
    paddle_center = paddle_x + paddle_width / 2
    delta_x = predicted_x - paddle_center
    
    if abs(delta_x) < env.PADDLE_SPEED:
        a0 = 0
    elif delta_x < 0:
        a0 = 3
    else:
        a0 = 4
        
    return [a0, 0, 0]