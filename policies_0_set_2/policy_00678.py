def policy(env):
    # Strategy: Track the ball's predicted x-position when it reaches the paddle's y-level, moving to intercept it.
    # This maximizes deflections (+1 reward) while minimizing unnecessary movement penalties (-0.02 when ball is moving away).
    if env.ball_vel.y <= 0:  # Ball moving away - avoid movement penalty
        return [0, 0, 0]
    
    # Predict ball's x-position at paddle height
    time_to_reach = (env.paddle_pos.y - env.BALL_RADIUS - env.ball_pos.y) / env.ball_vel.y
    future_x = env.ball_pos.x + env.ball_vel.x * time_to_reach
    
    # Account for wall bounces by reflecting predicted position
    while future_x < env.BALL_RADIUS or future_x > env.SCREEN_WIDTH - env.BALL_RADIUS:
        if future_x < env.BALL_RADIUS:
            future_x = 2 * env.BALL_RADIUS - future_x
        else:
            future_x = 2 * (env.SCREEN_WIDTH - env.BALL_RADIUS) - future_x
    
    paddle_center = env.paddle_pos.x + env.PADDLE_WIDTH / 2
    if abs(paddle_center - future_x) < 10:  # Close enough - avoid oscillation
        return [0, 0, 0]
    return [4 if future_x > paddle_center else 3, 0, 0]