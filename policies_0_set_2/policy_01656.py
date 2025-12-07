def policy(env):
    # Strategy: Track the ball's x-coordinate to position the paddle for rebounds, launch immediately to start scoring, and avoid unnecessary movements to prevent time penalties.
    obs = env._get_observation()
    height, width, _ = obs.shape
    paddle_y = height - 30
    ball_attached = True
    ball_x, ball_y = None, None
    paddle_x = None

    # Locate paddle by scanning for white pixels near expected y-position
    for y in range(paddle_y - 5, paddle_y + 5):
        for x in range(width):
            r, g, b = obs[y, x]
            if r > 200 and g > 200 and b > 200:  # White paddle
                paddle_x = x if paddle_x is None else (paddle_x + x) / 2

    # Locate ball by scanning for green pixels; if not found, assume attached
    for y in range(height):
        for x in range(width):
            r, g, b = obs[y, x]
            if g > 200 and r < 100 and b < 100:  # Green ball
                ball_x, ball_y = x, y
                ball_attached = False
                break
        if ball_x is not None:
            break

    # If ball not found, it is attached to paddle
    if ball_x is None and paddle_x is not None:
        ball_x, ball_y = paddle_x, paddle_y - 7
        ball_attached = True

    # Determine action
    a1 = 1 if ball_attached else 0  # Launch if attached
    a2 = 0  # Secondary action unused

    if ball_x is not None and paddle_x is not None:
        if paddle_x < ball_x - 5:
            a0 = 4  # Move right
        elif paddle_x > ball_x + 5:
            a0 = 3  # Move left
        else:
            a0 = 0  # Hold position
    else:
        a0 = 0  # Default to no movement

    return [a0, a1, a2]