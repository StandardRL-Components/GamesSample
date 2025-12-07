def policy(env):
    # Strategy: Track ball's horizontal position when descending, center paddle otherwise. Launch immediately when possible.
    # Avoids complex prediction for robustness. Uses tolerance to prevent oscillation.
    a0, a1, a2 = 0, 0, 0
    
    # Launch ball if not launched and space wasn't held previously
    if not env.ball_launched and not env.prev_space_held:
        a1 = 1
    
    # Track ball horizontally when moving downward
    if env.ball_launched and env.ball_vel[1] > 0:
        target_x = env.ball_pos[0]
        # Move paddle toward ball's x-position with tolerance
        if env.paddle.centerx < target_x - 10:
            a0 = 4
        elif env.paddle.centerx > target_x + 10:
            a0 = 3
    else:
        # Center paddle when ball moving upward or not launched
        center = env.WIDTH / 2
        if env.paddle.centerx < center - 5:
            a0 = 4
        elif env.paddle.centerx > center + 5:
            a0 = 3
    
    return [a0, a1, a2]