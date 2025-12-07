def policy(env):
    # Strategy: Launch ball immediately, then track its x-position to keep paddle aligned for rebounds.
    # This maximizes reward by maintaining ball in play (avoiding penalties) and breaking blocks efficiently.
    if env.ball_attached:
        return [0, 1, 0]  # Launch ball with primary action
    else:
        ball_x = env.ball.centerx
        paddle_x = env.paddle.centerx
        if paddle_x < ball_x - 10:
            return [4, 0, 0]  # Move right to align with ball
        elif paddle_x > ball_x + 10:
            return [3, 0, 0]  # Move left to align with ball
        else:
            return [0, 0, 0]  # Hold position when aligned