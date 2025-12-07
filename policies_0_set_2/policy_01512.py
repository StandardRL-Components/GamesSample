def policy(env):
    """
    Strategy: Launch ball immediately, then track its x-position to align paddle.
    Maximizes reward by keeping ball in play to break blocks while avoiding stillness penalties.
    """
    obs = env._get_observation()
    
    # Check if ball is launched by scanning for "PRESS SPACE" text near (320,300)
    text_region = [(x, y) for x in range(310, 330) for y in range(295, 305)]
    text_detected = any(all(obs[y, x] > 200) for x, y in text_region if 0 <= x < 640 and 0 <= y < 400)
    
    if text_detected:
        return [0, 1, 0]  # Launch ball immediately
    
    # Find ball centroid using yellow color threshold (R,G>200, B<50)
    ball_pixels = []
    for y in range(0, 400, 4):
        for x in range(0, 640, 4):
            r, g, b = obs[y, x]
            if r > 200 and g > 200 and b < 50:
                ball_pixels.append((x, y))
    
    if not ball_pixels:
        return [0, 0, 0]  # Ball not found
    
    ball_x = sum(p[0] for p in ball_pixels) // len(ball_pixels)
    
    # Find paddle centroid using white color threshold (R,G,B>200) in bottom region
    paddle_pixels = []
    for y in range(350, 390, 2):
        for x in range(0, 640, 2):
            r, g, b = obs[y, x]
            if r > 200 and g > 200 and b > 200:
                paddle_pixels.append(x)
    
    paddle_x = sum(paddle_pixels) // len(paddle_pixels) if paddle_pixels else 320
    
    # Move paddle toward ball's x-position
    if ball_x < paddle_x - 5:
        return [3, 0, 0]  # Move left
    elif ball_x > paddle_x + 5:
        return [4, 0, 0]  # Move right
    return [0, 0, 0]  # Hold position