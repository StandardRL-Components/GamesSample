def policy(env):
    # Strategy: Track the ball's x-position and move the paddle to intercept it when the ball is moving downward.
    # Avoid movement when ball is moving upward to prevent penalty. Secondary actions are unused in this environment.
    obs = env._get_observation()
    ball_y, ball_x = None, None
    paddle_x = None
    
    # Find ball (yellow pixels) and paddle (white pixels) by scanning observation
    for y in range(env.HEIGHT):
        for x in range(env.WIDTH):
            r, g, b = obs[y, x]
            # Ball is yellow (high R and G, low B)
            if r > 200 and g > 200 and b < 100:
                ball_y, ball_x = y, x
            # Paddle is white (high RGB)
            if r > 250 and g > 250 and b > 250 and y > env.HEIGHT - 20:
                paddle_x = x if paddle_x is None else paddle_x  # Keep first found position
    
    # Default to center if unable to detect
    if ball_x is None:
        ball_x = env.WIDTH // 2
    if paddle_x is None:
        paddle_x = env.WIDTH // 2
    
    # Move only when ball is below mid-screen (likely moving downward)
    if ball_y > env.HEIGHT // 2:
        if paddle_x < ball_x - 5:
            return [4, 0, 0]  # Move right
        elif paddle_x > ball_x + 5:
            return [3, 0, 0]  # Move left
    return [0, 0, 0]  # No movement