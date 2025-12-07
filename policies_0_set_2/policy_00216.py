def policy(env):
    # Strategy: Track ball's x-position when moving down to align paddle for bounce. 
    # Launch ball immediately if not launched. Move to center when ball is moving up.
    # This maximizes blocks broken by keeping ball in play and aiming for efficient bounces.
    
    # Read current state
    ball_launched = env.ball_launched
    ball_pos = env.ball_pos
    ball_vel = env.ball_vel
    paddle = env.paddle
    screen_width = env.SCREEN_WIDTH
    paddle_width = env.PADDLE_WIDTH
    
    # Determine movement action
    if not ball_launched:
        # Center paddle before launch
        center_x = (screen_width - paddle_width) / 2
        if paddle.x < center_x - 5:
            movement = 4  # Right
        elif paddle.x > center_x + 5:
            movement = 3  # Left
        else:
            movement = 0  # No movement
    else:
        if ball_vel[1] > 0:  # Ball moving downward
            # Predict where ball will hit paddle level
            time_to_paddle = (paddle.top - ball_pos[1]) / ball_vel[1]
            predicted_x = ball_pos[0] + ball_vel[0] * time_to_paddle
            target_x = max(0, min(screen_width - paddle_width, predicted_x - paddle_width/2))
            
            # Move paddle toward predicted position
            if paddle.x < target_x - 5:
                movement = 4  # Right
            elif paddle.x > target_x + 5:
                movement = 3  # Left
            else:
                movement = 0  # No movement
        else:
            # Ball moving upward - move to center
            center_x = (screen_width - paddle_width) / 2
            if paddle.x < center_x - 5:
                movement = 4  # Right
            elif paddle.x > center_x + 5:
                movement = 3  # Left
            else:
                movement = 0  # No movement
    
    # Always try to launch ball if not launched
    space_held = 1 if not ball_launched else 0
    
    # Shift not used in this environment
    shift_held = 0
    
    return [movement, space_held, shift_held]