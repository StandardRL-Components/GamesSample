def policy(env):
    """
    Maximizes reward by tracking the ball's projected landing position while avoiding safe-zone penalties.
    Uses predictive positioning based on ball trajectory and only enters safe zone when necessary to hit the ball.
    """
    if env.game_over:
        return [0, 0, 0]
    
    # Calculate ball's projected x-position when it reaches paddle height
    if env.ball_vel.y > 0:  # Ball moving downward
        time_to_paddle = (env.paddle.top - env.ball_pos.y) / env.ball_vel.y
        projected_x = env.ball_pos.x + env.ball_vel.x * time_to_paddle
        
        # Account for wall bounces in projection
        while projected_x < 0 or projected_x > env.WIDTH:
            if projected_x < 0:
                projected_x = -projected_x
            else:
                projected_x = 2 * env.WIDTH - projected_x
    else:
        projected_x = env.ball_pos.x
    
    # Safe zone boundaries
    safe_left = env.WIDTH * 0.4
    safe_right = env.WIDTH * 0.6
    is_approaching = env.ball_vel.y > 0 and env.ball_pos.y > env.HEIGHT * 0.7
    
    # Avoid safe zone penalty when ball is approaching
    if is_approaching and safe_left <= env.paddle.centerx <= safe_right:
        # Move to nearest safe zone edge
        if abs(env.paddle.centerx - safe_left) < abs(env.paddle.centerx - safe_right):
            target_x = safe_left - 1
        else:
            target_x = safe_right + 1
    else:
        target_x = projected_x
    
    # Move paddle toward target position
    if abs(target_x - env.paddle.centerx) < 5:
        return [0, 0, 0]
    elif target_x < env.paddle.centerx:
        return [3, 0, 0]
    else:
        return [4, 0, 0]