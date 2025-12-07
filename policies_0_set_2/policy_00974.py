def policy(env):
    # Strategy: Launch ball immediately, then track predicted landing position using physics.
    # This maximizes reward by keeping ball in play and breaking blocks efficiently.
    if not env.ball_launched:
        return [0, 1, 0]  # Launch ball with space
    
    if env.ball_vy > 0:  # Ball moving downward
        left_bound = env.BALL_RADIUS
        right_bound = env.WIDTH - env.BALL_RADIUS
        range_x = right_bound - left_bound
        target_y = env.HEIGHT - env.PADDLE_HEIGHT - 5 - env.BALL_RADIUS
        
        if env.ball_y >= target_y:
            predicted_x = env.ball_x
        else:
            t = (target_y - env.ball_y) / env.ball_vy
            total_dx = env.ball_vx * t
            x0 = env.ball_x - left_bound
            x_unfolded = x0 + total_dx
            x_normalized = x_unfolded % (2 * range_x)
            if x_normalized > range_x:
                x_normalized = 2 * range_x - x_normalized
            predicted_x = x_normalized + left_bound
        
        if env.paddle_x < predicted_x - 5:
            movement = 4  # Right
        elif env.paddle_x > predicted_x + 5:
            movement = 3  # Left
        else:
            movement = 0  # None
    else:  # Ball moving upward
        center_x = env.WIDTH / 2
        if env.paddle_x < center_x - 5:
            movement = 4  # Right
        elif env.paddle_x > center_x + 5:
            movement = 3  # Left
        else:
            movement = 0  # None
    
    return [movement, 0, 0]