def policy(env):
    # Strategy: Track ball's predicted x-position when it reaches paddle height, accounting for wall bounces.
    # Move paddle to intercept, prioritizing precise alignment to maximize block breaks and prevent ball loss.
    ball_x, ball_y = env.ball['pos']
    vx, vy = env.ball['vel']
    paddle_center = env.paddle.x + env.PADDLE_WIDTH / 2
    
    if vy > 0:  # Ball moving downward
        # Predict time to reach paddle height
        time_to_paddle = (env.paddle.top - env.BALL_RADIUS - ball_y) / vy
        # Project x-position with wall bounces
        projected_x = ball_x + vx * time_to_paddle
        segment = env.WIDTH - 2 * env.BALL_RADIUS
        if segment > 0:
            normalized = (projected_x - env.BALL_RADIUS) % (2 * segment)
            if normalized > segment:
                normalized = 2 * segment - normalized
            target_x = normalized + env.BALL_RADIUS
        else:
            target_x = env.WIDTH / 2
    else:
        target_x = env.WIDTH / 2  # Default to center if ball moving upward
    
    # Move paddle toward target
    if target_x < paddle_center - 5:
        movement = 3  # Left
    elif target_x > paddle_center + 5:
        movement = 4  # Right
    else:
        movement = 0  # None
    
    return [movement, 0, 0]