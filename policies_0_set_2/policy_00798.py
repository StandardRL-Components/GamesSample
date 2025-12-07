def policy(env):
    # Strategy: Prioritize launching the ball, then move paddle to intercept ball's trajectory
    # when moving downward. Aim for center of paddle to minimize misses, but slight edge hits
    # for bonus rewards when safe. Always return valid [a0, a1, a2] actions.
    if env.game_over or env.game_won:
        return [0, 0, 0]
    
    if env.ball_attached:
        return [0, 1, 0]  # Launch ball immediately when attached
    
    paddle_center = env.paddle.x + env.paddle.width / 2
    ball_x, ball_y = env.ball_pos
    ball_vx, ball_vy = env.ball_vel
    
    # Predict ball position when it reaches paddle height
    if ball_vy > 0:  # Ball moving downward
        time_to_paddle = (env.paddle.top - ball_y) / ball_vy
        predicted_x = ball_x + ball_vx * time_to_paddle
        # Account for wall bounces
        while predicted_x < 0 or predicted_x > env.WIDTH:
            predicted_x = 2 * (env.WIDTH if predicted_x > env.WIDTH else 0) - predicted_x
        target_x = predicted_x
    else:
        target_x = env.WIDTH / 2  # Center paddle when ball moving upward
    
    # Move paddle toward target position
    if paddle_center < target_x - 5:
        movement = 4  # Right
    elif paddle_center > target_x + 5:
        movement = 3  # Left
    else:
        movement = 0  # None
    
    return [movement, 0, 0]