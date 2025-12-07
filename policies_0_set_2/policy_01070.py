def policy(env):
    # Strategy: Track ball's x-position and move paddle to intercept. Launch ball immediately if not in play.
    # Maximizes reward by breaking blocks (immediate reward) and preventing ball loss (large penalty).
    obs = env._get_observation()
    height, width, _ = obs.shape
    white_pixels = []
    for y in range(50, 350, 2):
        for x in range(10, 630, 2):
            r, g, b = obs[y, x]
            if r > 250 and g > 250 and b > 250:
                white_pixels.append((x, y))
    
    if not white_pixels:
        return [0, 1, 0]
    
    ball_x = sum(x for x, y in white_pixels) // len(white_pixels)
    
    paddle_pixels = []
    for y in range(380, 390, 2):
        for x in range(10, 630, 2):
            r, g, b = obs[y, x]
            if r > 210 and g > 210 and b > 240:
                paddle_pixels.append(x)
                
    paddle_center = sum(paddle_pixels) // len(paddle_pixels) if paddle_pixels else width // 2
    
    if paddle_center < ball_x - 5:
        movement = 4
    elif paddle_center > ball_x + 5:
        movement = 3
    else:
        movement = 0
        
    return [movement, 0, 0]