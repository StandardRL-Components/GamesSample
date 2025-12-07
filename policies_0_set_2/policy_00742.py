def policy(env):
    # Strategy: Launch ball immediately if attached. Otherwise, predict ball's landing position based on velocity
    # and move paddle to intercept. Prioritize catching falling balls to prevent loss and maximize block breaks.
    if env.ball_attached:
        return [0, 1, 0]  # Launch ball without moving
    paddle_center = env.paddle.centerx
    if env.ball_vel[1] > 0:  # Ball is falling
        time_to_paddle = (env.paddle.top - env.ball_pos[1]) / env.ball_vel[1]
        future_x = env.ball_pos[0] + env.ball_vel[0] * time_to_paddle
        future_x = max(env.WALL_THICKNESS, min(future_x, env.SCREEN_WIDTH - env.WALL_THICKNESS))
        if future_x < paddle_center - 5:
            return [3, 0, 0]  # Move left
        elif future_x > paddle_center + 5:
            return [4, 0, 0]  # Move right
        else:
            return [0, 0, 0]  # Stay centered
    else:  # Ball is rising
        screen_center = env.SCREEN_WIDTH / 2
        if paddle_center < screen_center - 5:
            return [4, 0, 0]  # Move toward center
        elif paddle_center > screen_center + 5:
            return [3, 0, 0]  # Move toward center
        else:
            return [0, 0, 0]  # Already centered