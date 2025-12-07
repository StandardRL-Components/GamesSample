def policy(env):
    # Strategy: Launch ball immediately, then move paddle to intercept ball's predicted landing point.
    # This maximizes block breaks by keeping ball in play and targeting high-value blocks via bounces.
    if env.game_over:
        return [0, 0, 0]
    if env.ball_attached:
        return [0, 1, 0]
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    paddle_x = env.paddle.centerx
    if ball_vy > 0.1:
        time_to_paddle = (env.paddle.top - ball_y) / ball_vy
        predicted_x = ball_x + ball_vx * time_to_paddle
        predicted_x = max(50, min(590, predicted_x))
        if paddle_x < predicted_x - 5:
            return [4, 0, 0]
        elif paddle_x > predicted_x + 5:
            return [3, 0, 0]
    return [0, 0, 0]