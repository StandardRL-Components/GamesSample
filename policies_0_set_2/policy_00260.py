def policy(env):
    # Strategy: Track ball's x-position when moving down, center paddle when moving up.
    # Launch immediately when ball is on paddle. Maximizes blocks broken and minimizes ball loss.
    if env.ball_on_paddle:
        return [0, 1, 0]  # No movement, launch ball
    else:
        if env.ball_vel[1] > 0:  # Ball moving downward
            target_x = env.ball.centerx + env.ball_vel[0] * ((env.paddle.top - env.ball.centery) / env.ball_vel[1])
            # Account for wall bounces
            while target_x < 0 or target_x > env.WIDTH:
                if target_x < 0:
                    target_x = -target_x
                else:
                    target_x = 2 * env.WIDTH - target_x
            if env.paddle.centerx < target_x - 5:
                return [4, 0, 0]  # Move right
            elif env.paddle.centerx > target_x + 5:
                return [3, 0, 0]  # Move left
            else:
                return [0, 0, 0]  # Hold position
        else:  # Ball moving upward
            center = env.WIDTH / 2
            if env.paddle.centerx < center - 5:
                return [4, 0, 0]  # Move right toward center
            elif env.paddle.centerx > center + 5:
                return [3, 0, 0]  # Move left toward center
            else:
                return [0, 0, 0]  # Hold center