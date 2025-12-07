def policy(env):
    """
    Breakout strategy: Track the ball's x-position and move the paddle to intercept it.
    Since the environment only uses the first action component (paddle movement),
    we focus on moving left/right to align the paddle with the ball's projected landing point.
    This maximizes ball returns and block breaks while minimizing life loss.
    """
    obs = env._get_observation()
    height, width, _ = obs.shape
    
    # Find paddle position (light blue bar at bottom)
    paddle_pixels = []
    for y in range(360, height):
        for x in range(width):
            r, g, b = obs[y, x]
            if 210 < r < 230 and 210 < g < 230 and b > 250:
                paddle_pixels.append(x)
    paddle_x = width // 2
    if paddle_pixels:
        sorted_paddle = sorted(paddle_pixels)
        n = len(sorted_paddle)
        mid = n // 2
        paddle_x = sorted_paddle[mid] if n % 2 == 1 else (sorted_paddle[mid-1] + sorted_paddle[mid]) // 2

    # Find ball position (red circle)
    red_pixels = []
    for y in range(0, 360):
        for x in range(width):
            r, g, b = obs[y, x]
            if r > 200 and g < 100 and b < 100:
                red_pixels.append(x)
    if not red_pixels:
        return [0, 0, 0]
    
    sorted_red = sorted(red_pixels)
    n_red = len(sorted_red)
    mid_red = n_red // 2
    ball_x = sorted_red[mid_red] if n_red % 2 == 1 else (sorted_red[mid_red-1] + sorted_red[mid_red]) // 2

    # Move paddle toward ball with tolerance to prevent oscillation
    if ball_x < paddle_x - 10:
        return [3, 0, 0]
    elif ball_x > paddle_x + 10:
        return [4, 0, 0]
    else:
        return [0, 0, 0]