def policy(env):
    # This policy maximizes score by launching the ball immediately and then tracking its predicted
    # landing position when moving downward, while centering the paddle when the ball is moving upward.
    # This balances immediate block breaking with defensive positioning to avoid losing lives.
    if env.ball_attached:
        center_target = env.WIDTH // 2
        if env.paddle.centerx < center_target - 5:
            a0 = 4  # Move right toward center
        elif env.paddle.centerx > center_target + 5:
            a0 = 3  # Move left toward center
        else:
            a0 = 0  # Already centered
        a1 = 1  # Launch ball
        a2 = 0
    else:
        if env.ball_vel[1] > 0:  # Ball moving downward
            time_to_paddle = (env.paddle.top - (env.ball_pos[1] + env.BALL_RADIUS)) / env.ball_vel[1]
            target_x = env.ball_pos[0] + env.ball_vel[0] * time_to_paddle
            target_x = max(0, min(env.WIDTH, target_x))
        else:
            target_x = env.WIDTH // 2  # Center when ball moving upward
        
        if target_x < env.paddle.centerx - 5:
            a0 = 3  # Move left
        elif target_x > env.paddle.centerx + 5:
            a0 = 4  # Move right
        else:
            a0 = 0  # Hold position
        a1 = 0
        a2 = 0
    return [a0, a1, a2]