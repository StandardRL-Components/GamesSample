def policy(env):
    # Strategy: Track ball position when moving downward to align paddle for rebounds.
    # Launch ball immediately when attached. Prioritize preventing ball loss by
    # positioning paddle under predicted ball trajectory when falling.
    if env.game_over:
        return [0, 0, 0]
    
    if env.ball_attached:
        return [0, 1, 0]
    
    if env.ball_vel[1] > 0 and env.ball_pos[1] < env.paddle.top:
        target_x = env.ball_pos[0] - env.PADDLE_WIDTH / 2
        target_x = max(0, min(env.SCREEN_WIDTH - env.PADDLE_WIDTH, target_x))
        current_x = env.paddle.x
        if current_x < target_x - env.PADDLE_SPEED:
            return [4, 0, 0]
        elif current_x > target_x + env.PADDLE_SPEED:
            return [3, 0, 0]
    
    return [0, 0, 0]