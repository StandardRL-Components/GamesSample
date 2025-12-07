def policy(env):
    # Track the ball's predicted x-position when moving down to maximize hits and block breaks.
    # When moving up, align with current ball x to prepare. Avoid oscillation with a tolerance.
    paddle_center = env.paddle.x + env.paddle.width / 2
    if env.ball_vel[1] > 0:  # Ball moving down: predict landing position
        time_to_reach = (env.paddle.top - env.ball_pos[1]) / env.ball_vel[1]
        target_x = env.ball_pos[0] + env.ball_vel[0] * time_to_reach
        target_x = max(env.BALL_RADIUS, min(target_x, env.WIDTH - env.BALL_RADIUS))
    else:  # Ball moving up: track current x
        target_x = env.ball_pos[0]
    
    if paddle_center < target_x - 5:
        return [4, 0, 0]  # Move right
    elif paddle_center > target_x + 5:
        return [3, 0, 0]  # Move left
    else:
        return [0, 0, 0]  # No movement