def policy(env):
    # Strategy: Track ball x-position and move paddle to intercept. Launch ball immediately when on paddle.
    # This maximizes reward by preventing life loss and breaking blocks quickly.
    obs = env._get_observation()
    h, w = obs.shape[:2]
    
    # Find paddle position (light blue at bottom)
    paddle_y = h - 20
    paddle_pixels = []
    for x in range(0, w, 5):
        if all(obs[paddle_y, x] > [200, 200, 200]):
            paddle_pixels.append(x)
    paddle_x = sum(paddle_pixels) / len(paddle_pixels) if paddle_pixels else w/2
    
    # Find ball position (white pixel search)
    ball_pos = None
    for y in range(h//2):  # Only search upper half
        for x in range(w):
            if all(obs[y, x] > [250, 250, 250]):
                ball_pos = x
                break
        if ball_pos is not None:
            break
    
    # Determine action
    action = [0, 0, 0]
    if ball_pos is None:  # Ball likely on paddle
        action[1] = 1  # Launch ball
    else:
        if ball_pos < paddle_x - 15:
            action[0] = 3  # Move left
        elif ball_pos > paddle_x + 15:
            action[0] = 4  # Move right
    
    return action