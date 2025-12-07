def policy(env):
    # Strategy: Align paddle with ball's x-coordinate to maximize bounces and block breaks.
    # Launch ball immediately when available. Ignore beat timing due to complexity.
    if env.ball_on_paddle:
        return [0, 1, 0]  # No movement, launch ball
    else:
        ball_x = env.ball_pos[0]
        paddle_x = env.paddle_rect.centerx
        if paddle_x < ball_x - 5:
            return [4, 0, 0]  # Move right
        elif paddle_x > ball_x + 5:
            return [3, 0, 0]  # Move left
        else:
            return [0, 0, 0]  # No movement