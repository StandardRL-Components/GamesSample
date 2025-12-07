def policy(env):
    # Strategy: Track the ball with the shortest time to reach the paddle and move to intercept its predicted position after accounting for wall bounces.
    # Maximizes reward by preventing misses (avoiding -10 penalty) and maximizing bounces (+1 each) to achieve survival bonus (+100).
    candidate_ball = None
    min_time = float('inf')
    paddle_center_x = env.paddle.x + env.PADDLE_WIDTH / 2
    for ball in env.balls:
        if ball['pos'].y > env.PADDLE_Y and ball['vel'].y > 0:
            continue
        if ball['vel'].y > 0:
            t = (env.PADDLE_Y - ball['pos'].y) / ball['vel'].y
        else:
            t = (-ball['pos'].y) / ball['vel'].y + env.PADDLE_Y / abs(ball['vel'].y)
        if t < min_time:
            min_time = t
            candidate_ball = ball
    if candidate_ball is None:
        return [0, 0, 0]
    future_x = candidate_ball['pos'].x + candidate_ball['vel'].x * min_time
    future_x = future_x % (2 * env.SCREEN_WIDTH)
    if future_x > env.SCREEN_WIDTH:
        future_x = 2 * env.SCREEN_WIDTH - future_x
    target_center = future_x
    if target_center < paddle_center_x - 5:
        return [3, 0, 0]
    elif target_center > paddle_center_x + 5:
        return [4, 0, 0]
    else:
        return [0, 0, 0]