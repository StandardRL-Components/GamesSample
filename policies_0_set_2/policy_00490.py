def policy(env):
    # Strategy: Track the ball's predicted landing point to align the paddle, ensuring maximum blocks are broken without losing lives.
    # Launch the ball immediately when on the paddle to minimize step penalty and maximize breaking efficiency.
    threshold = 5  # Pixel threshold to avoid unnecessary movement oscillations

    if env.on_paddle:
        # Launch ball and center paddle for better coverage
        center_x = env.WIDTH // 2
        if abs(env.paddle.centerx - center_x) < threshold:
            move_action = 0
        else:
            move_action = 4 if env.paddle.centerx < center_x else 3
        return [move_action, 1, 0]

    # Predict ball's future x-position at paddle height
    paddle_y = env.HEIGHT - env.PADDLE_HEIGHT - 10
    target_y = paddle_y - env.BALL_RADIUS
    current_y = env.ball_pos[1]
    current_vy = env.ball_vel[1]

    if current_vy == 0:
        future_x = env.ball_pos[0]
    else:
        if current_vy > 0:
            time_to_target = (target_y - current_y) / current_vy
        else:
            time_to_top = (env.BALL_RADIUS - current_y) / current_vy
            time_after_bounce = (target_y - env.BALL_RADIUS) / (-current_vy)
            time_to_target = time_to_top + time_after_bounce

        current_x = env.ball_pos[0]
        current_vx = env.ball_vel[0]
        period = 2 * (env.WIDTH - 2 * env.BALL_RADIUS)
        if period == 0:
            future_x = current_x
        else:
            unbounded_x = current_x + current_vx * time_to_target
            normalized_x = (unbounded_x - env.BALL_RADIUS) % period
            if normalized_x < 0:
                normalized_x += period
            if normalized_x > period / 2:
                normalized_x = period - normalized_x
            future_x = normalized_x + env.BALL_RADIUS

    # Move paddle toward predicted position
    if abs(env.paddle.centerx - future_x) < threshold:
        move_action = 0
    else:
        move_action = 4 if env.paddle.centerx < future_x else 3

    return [move_action, 0, 0]