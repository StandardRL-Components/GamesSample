def policy(env):
    """
    Breakout strategy: Launch ball immediately, then track ball's x-position when descending.
    Maximizes brick breaks (reward) by keeping ball in play and directing it toward bricks.
    Avoids losing lives by catching descending ball with paddle.
    """
    if env.game_over:
        return [0, 0, 0]
    
    if not env.ball_launched:
        return [0, 1, 0]
    
    paddle_center = env.paddle_rect.centerx
    if env.ball_vel.y > 0:  # Ball descending - track its x-position
        ball_x = env.ball_pos.x
        if ball_x < paddle_center - 10:
            return [3, 0, 0]
        elif ball_x > paddle_center + 10:
            return [4, 0, 0]
        else:
            return [0, 0, 0]
    else:  # Ball ascending - center paddle for better coverage
        screen_center = env.SCREEN_WIDTH / 2
        if paddle_center < screen_center - 5:
            return [4, 0, 0]
        elif paddle_center > screen_center + 5:
            return [3, 0, 0]
        else:
            return [0, 0, 0]