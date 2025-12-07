def policy(env):
    # Strategy: Predict ball's future x-position when it reaches paddle height and align paddle center.
    # This maximizes hits and combos while minimizing misses, prioritizing immediate rewards and block breaks.
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    paddle_top = env.paddle.top
    paddle_width = env.paddle.width
    current_paddle_center = env.paddle.x + paddle_width / 2

    if ball_vy > 0:  # Ball moving downward
        time_to_reach = (paddle_top - ball_y) / ball_vy
        future_x = ball_x + ball_vx * time_to_reach
        future_x = max(0, min(env.WIDTH, future_x))  # Clamp to screen bounds
        target_center = future_x
    else:  # Ball moving upward, center paddle as default
        target_center = env.WIDTH / 2

    if current_paddle_center < target_center - 5:
        movement = 4  # Right
    elif current_paddle_center > target_center + 5:
        movement = 3  # Left
    else:
        movement = 0  # None

    return [movement, 0, 0]  # a1 and a2 unused in this environment