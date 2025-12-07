def policy(env):
    # This policy tracks the ball's horizontal position and moves the paddle to align with it.
    # It launches the ball when not in play. Maximizes reward by preventing ball loss and breaking bricks.
    obs = env._get_observation()
    # Find ball by detecting green pixels (RGB: ~[100,255,100])
    green = obs[:,:,1]
    red = obs[:,:,0]
    blue = obs[:,:,2]
    ball_mask = (green > 200) & (red < 100) & (blue < 100)
    ball_coords = np.where(ball_mask)
    if len(ball_coords[0]) > 0:
        ball_y = np.mean(ball_coords[0])
        ball_x = np.mean(ball_coords[1])
        ball_launched = ball_y < 370  # Ball is above paddle area
    else:
        ball_launched = False
        ball_x = None

    # Find paddle in bottom rows (RGB: ~[220,220,255])
    bottom_slice = obs[380:400, :, :]
    blue_b = bottom_slice[:,:,2]
    red_b = bottom_slice[:,:,0]
    green_b = bottom_slice[:,:,1]
    paddle_mask = (blue_b > 200) & (red_b > 200) & (red_b < 240) & (green_b > 200) & (green_b < 240)
    paddle_coords = np.where(paddle_mask)
    paddle_x = np.mean(paddle_coords[1]) if len(paddle_coords[1]) > 0 else None

    # Move paddle toward ball if launched and visible
    if ball_launched and ball_x is not None and paddle_x is not None:
        tolerance = 5
        if paddle_x < ball_x - tolerance:
            move = 4  # Right
        elif paddle_x > ball_x + tolerance:
            move = 3  # Left
        else:
            move = 0
    else:
        move = 0

    # Launch ball if not launched
    launch = 1 if not ball_launched else 0

    return [move, launch, 0]