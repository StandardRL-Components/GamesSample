def policy(env):
    # Strategy: Track ball position and velocity to intercept with paddle. Launch immediately when attached.
    # Prioritize keeping ball in play to break bricks and maximize score with minimal steps.
    if env.ball_attached:
        # Launch ball immediately and move paddle toward center to prepare for return
        target_x = env.WIDTH // 2
        current_x = env.paddle.centerx
        if current_x < target_x - 5:
            movement = 4  # Right
        elif current_x > target_x + 5:
            movement = 3  # Left
        else:
            movement = 0  # None
        return [movement, 1, 0]  # Launch ball (a1=1)
    else:
        # Predict ball trajectory and move paddle to intercept when ball is moving down
        if env.ball_vel[1] > 0:  # Ball moving downward
            # Estimate time to reach paddle height and predict x position
            time_to_paddle = (env.paddle.top - env.ball_pos[1]) / env.ball_vel[1]
            future_x = env.ball_pos[0] + env.ball_vel[0] * time_to_paddle
            # Clamp prediction to valid range and adjust for paddle width
            target_x = min(max(future_x, env.PADDLE_WIDTH / 2), env.WIDTH - env.PADDLE_WIDTH / 2)
            current_x = env.paddle.centerx
            if current_x < target_x - 5:
                movement = 4  # Right
            elif current_x > target_x + 5:
                movement = 3  # Left
            else:
                movement = 0  # None
        else:
            # Ball moving upward, move to center to prepare
            current_x = env.paddle.centerx
            target_x = env.WIDTH // 2
            if current_x < target_x - 5:
                movement = 4  # Right
            elif current_x > target_x + 5:
                movement = 3  # Left
            else:
                movement = 0  # None
        return [movement, 0, 0]  # No launch (a1=0), no secondary action (a2=0)