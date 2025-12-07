def policy(env):
    # Strategy: Maximize survival and block breaking by predicting ball trajectory and aligning paddle.
    # Prioritize launching ball immediately, then move paddle to intercept ball's predicted landing point.
    # Use wall bounce simulation for accurate prediction and a deadzone to prevent oscillation.
    if env.game_state == "BALL_HELD":
        return [0, 1, 0]  # Launch ball immediately
    elif env.game_state == "PLAYING":
        if env.ball_vel.y > 0:  # Ball moving downward
            time_to_paddle = (env.paddle.top - env.ball_pos.y) / env.ball_vel.y
            future_x = env.ball_pos.x + env.ball_vel.x * time_to_paddle
            # Simulate wall bounces
            while future_x < 0 or future_x > env.WIDTH:
                if future_x < 0:
                    future_x = -future_x
                else:
                    future_x = 2 * env.WIDTH - future_x
            target_x = future_x
        else:
            target_x = env.WIDTH / 2  # Default to center if ball moving upward
        
        paddle_center = env.paddle.centerx
        if abs(paddle_center - target_x) < 10:
            move = 0
        elif paddle_center < target_x:
            move = 4
        else:
            move = 3
        return [move, 0, 0]
    else:
        return [0, 0, 0]  # No action for terminal states