def policy(env):
    # Strategy: Predict ball's landing position using current trajectory and move paddle to intercept.
    # Launch immediately to avoid penalty. Prioritize intercepting downward-moving balls to prevent loss.
    a0, a1, a2 = 0, 0, 0  # Default: no movement, no action
    
    # Launch ball if not already launched
    if not env.ball['launched']:
        a1 = 1
    else:
        # Only track ball when moving downward to avoid unnecessary movement
        if env.ball['vel'].y > 0:
            # Predict future x-position when ball reaches paddle height
            time_to_paddle = (env.paddle.top - env.ball['pos'].y) / env.ball['vel'].y
            predicted_x = env.ball['pos'].x + env.ball['vel'].x * time_to_paddle
            
            # Clamp prediction to screen bounds
            predicted_x = max(env.ball['radius'], min(env.SCREEN_WIDTH - env.ball['radius'], predicted_x))
            
            # Move paddle toward predicted position with deadzone
            paddle_center = env.paddle.centerx
            if predicted_x < paddle_center - 10:
                a0 = 3  # Left
            elif predicted_x > paddle_center + 10:
                a0 = 4  # Right
    
    return [a0, a1, a2]