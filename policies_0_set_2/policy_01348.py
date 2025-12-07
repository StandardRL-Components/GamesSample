def policy(env):
    # Strategy: Move catcher to align with the lowest (closest) fruit to maximize immediate rewards.
    # Prioritize fruits with higher value (blue=3, green=2, red=1) when multiple fruits are at similar heights.
    # Always return valid action with a1=0 and a2=0 since only movement (a0) affects gameplay.
    
    # Find catcher position by scanning bottom row for catcher color
    obs = env._get_observation()
    catcher_colors = [(255, 200, 0), (255, 255, 255)]  # Normal and flash colors
    catcher_x = None
    for x in range(env.WIDTH):
        pixel = obs[env.HEIGHT - 10, x]  # Sample middle of catcher area
        if tuple(pixel) in catcher_colors:
            catcher_x = x
            break
    current_col = catcher_x // env.GRID_CELL_WIDTH if catcher_x is not None else env.GRID_COLS // 2

    # Find lowest fruit in each column
    best_y = [-1] * env.GRID_COLS
    best_value = [0] * env.GRID_COLS
    fruit_colors = [(255, 50, 50), (50, 255, 50), (80, 150, 255)]  # R, G, B
    for y in range(env.HEIGHT - 20, 0, -1):  # Scan from bottom up
        for x in range(env.WIDTH):
            pixel = tuple(obs[y, x])
            if pixel in fruit_colors:
                col = min(x // env.GRID_CELL_WIDTH, env.GRID_COLS - 1)
                value = 3 if pixel == (80, 150, 255) else (2 if pixel == (50, 255, 50) else 1)
                if y > best_y[col] or (y == best_y[col] and value > best_value[col]):
                    best_y[col] = y
                    best_value[col] = value

    # Select target column with lowest fruit (highest y-value), prioritizing higher-value fruits
    target_col = current_col
    max_y = -1
    max_value = 0
    for col in range(env.GRID_COLS):
        if best_y[col] > max_y or (best_y[col] == max_y and best_value[col] > max_value):
            max_y = best_y[col]
            max_value = best_value[col]
            target_col = col

    # Move toward target column
    movement = 0  # No movement
    if target_col < current_col:
        movement = 3  # Left
    elif target_col > current_col:
        movement = 4  # Right

    return [movement, 0, 0]