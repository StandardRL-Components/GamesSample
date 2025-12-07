def policy(env):
    # Strategy: Track the ball's x-coordinate when moving downward to align paddle, maximizing bounces and block breaks.
    # Launch immediately when ball is attached. Prioritize ball tracking to prevent loss and maximize score.
    if env.ball_attached:
        return [0, 1, 0]  # No movement, launch ball
    else:
        if env.ball_vel[1] > 0:  # Ball moving downward
            # Predict future x-position at paddle height
            time_to_paddle = (env.paddle.top - env.ball_pos[1]) / env.ball_vel[1]
            future_x = env.ball_pos[0] + env.ball_vel[0] * time_to_paddle
            # Clamp to screen bounds and adjust for paddle width
            half_paddle = env.PADDLE_WIDTH / 2
            target_x = max(half_paddle, min(env.WIDTH - half_paddle, future_x))
        else:
            target_x = env.WIDTH / 2  # Default to center if ball moving upward
        
        # Move paddle toward target position
        current_x = env.paddle.centerx
        if current_x < target_x - 5:
            return [4, 0, 0]  # Move right
        elif current_x > target_x + 5:
            return [3, 0, 0]  # Move left
        else:
            return [0, 0, 0]  # Hold position