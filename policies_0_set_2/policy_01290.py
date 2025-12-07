def policy(env):
    # Strategy: Predict ball's future x-position when it reaches paddle's y-level and move paddle to intercept.
    # Maximizes reward by intercepting ball (high +reward) while minimizing unnecessary movement (-penalty).
    ball_pos, ball_vel, paddle = env.ball_pos, env.ball_vel, env.paddle
    if ball_vel.y > 0.001:  # Ball moving downward
        time_to_reach = (paddle.top - ball_pos.y) / ball_vel.y
        future_x = ball_pos.x + ball_vel.x * time_to_reach
        future_x = max(paddle.width/2, min(env.WIDTH - paddle.width/2, future_x))
        if paddle.centerx < future_x - 5:
            a0 = 4  # Right
        elif paddle.centerx > future_x + 5:
            a0 = 3  # Left
        else:
            a0 = 0  # None
    else:
        a0 = 0  # Wait if ball moving upward
    return [a0, 0, 0]