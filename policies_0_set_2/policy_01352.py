def policy(env):
    # Strategy: Track lowest visible blocks and align paddle to intercept them.
    # Prioritize catching blocks to maximize score and avoid life loss.
    # Use simple color thresholding to detect blocks and paddle from RGB observation.
    obs = env._get_observation().tolist()
    paddle_y = 375
    yellow_points = []
    for x in range(0, 640, 5):
        color = obs[paddle_y][x]
        r, g, b = color
        if r > 200 and g > 200 and b < 100:
            yellow_points.append(x)
    if yellow_points:
        paddle_center = (min(yellow_points) + max(yellow_points)) // 2
    else:
        paddle_center = 320

    block_found = None
    for y in range(375, 199, -10):
        for x in range(0, 640, 32):
            color = obs[y][x]
            r, g, b = color
            if (r > 200 and g < 100 and b < 100) or (r < 100 and g > 200 and b < 100) or (r < 100 and g < 200 and b > 100):
                block_found = (x, y)
                break
        if block_found:
            break

    if block_found:
        target_x = block_found[0] + 16
    else:
        target_x = 320

    if paddle_center < target_x - 5:
        movement = 4
    elif paddle_center > target_x + 5:
        movement = 3
    else:
        movement = 0

    return [movement, 0, 0]