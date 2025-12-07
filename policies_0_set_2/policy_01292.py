def policy(env):
    """
    Tracks the ball's predicted y-position when moving towards the player to align the paddle.
    Minimizes movement cost by only adjusting when necessary, and centers the paddle when
    the ball is moving away to prepare for the next return. Uses a tolerance to avoid oscillation.
    """
    # Read current game state
    ball_pos = env.ball_pos
    ball_vel = env.ball_vel
    paddle = env.player_paddle
    paddle_center = paddle.y + paddle.height / 2

    # Predict ball trajectory when moving towards player
    if ball_vel.x < 0 and ball_pos.x < env.WIDTH / 2:
        time_to_reach = (paddle.x - ball_pos.x) / ball_vel.x
        predicted_y = ball_pos.y + ball_vel.y * time_to_reach
        
        # Account for wall bounces
        while predicted_y < 0 or predicted_y > env.HEIGHT:
            if predicted_y < 0:
                predicted_y = -predicted_y
            else:
                predicted_y = 2 * env.HEIGHT - predicted_y
                
        target_y = predicted_y
    else:
        # Center paddle when ball is moving away
        target_y = env.HEIGHT / 2

    # Move paddle with tolerance to minimize unnecessary movement
    if paddle_center < target_y - 5:
        a0 = 2  # down
    elif paddle_center > target_y + 5:
        a0 = 1  # up
    else:
        a0 = 0  # none

    return [a0, 0, 0]