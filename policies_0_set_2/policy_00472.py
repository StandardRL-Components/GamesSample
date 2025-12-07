def policy(env):
    # Strategy: Track ball position and velocity to predict landing spot, moving paddle to intercept.
    # Prioritize launching ball immediately, then focus on keeping ball in play by aligning paddle with predicted bounce.
    # Use edge hits for bonus points when safe, but avoid missing the ball to prevent life loss.
    ball_launched = env.ball_launched
    ball_pos = env.ball_pos
    ball_vel = env.ball_vel
    paddle = env.paddle
    
    # Always try to launch ball if not already launched
    a1 = 1 if not ball_launched else 0
    
    # Calculate target paddle position based on ball trajectory
    if ball_launched and ball_vel[1] > 0:  # Ball moving downward
        # Predict where ball will hit paddle level
        time_to_paddle = (paddle.top - ball_pos[1]) / ball_vel[1]
        predicted_x = ball_pos[0] + ball_vel[0] * time_to_paddle
        
        # Clamp prediction to valid range
        min_x = 5 + paddle.width / 2
        max_x = env.WIDTH - 5 - paddle.width / 2
        target_x = max(min_x, min(predicted_x, max_x))
    else:
        # Center paddle when ball moving upward or not launched
        target_x = env.WIDTH / 2

    # Move paddle toward target position
    if paddle.centerx < target_x - 5:
        a0 = 4  # Right
    elif paddle.centerx > target_x + 5:
        a0 = 3  # Left
    else:
        a0 = 0  # None

    # Secondary action unused in this game
    a2 = 0

    return [a0, a1, a2]