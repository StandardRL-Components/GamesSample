def policy(env):
    # Prioritize selling harvested crops for immediate reward, then harvest mature crops for points, then plant seeds to maximize future yield.
    # Movement targets nearest empty or mature plot to minimize wasted steps and avoid oscillation.
    if env.harvested_crops > 0:
        return [0, 0, 1]  # Sell harvested crops for maximum coin gain
    
    r, c = env.selected_plot
    current_plot = env.plots[r][c]
    if current_plot['state'] == 'grown':
        return [0, 1, 0]  # Harvest mature crop
    if current_plot['state'] == 'empty':
        return [0, 1, 0]  # Plant seed
    
    # Find nearest empty or mature plot
    best_dist = float('inf')
    target_r, target_c = r, c
    for i in range(env.GRID_ROWS):
        for j in range(env.GRID_COLS):
            plot = env.plots[i][j]
            if plot['state'] in ['empty', 'grown']:
                dist = abs(i - r) + abs(j - c)
                if dist < best_dist:
                    best_dist = dist
                    target_r, target_c = i, j
    
    # Move toward target plot
    if target_r < r:
        return [1, 0, 0]  # Move up
    elif target_r > r:
        return [2, 0, 0]  # Move down
    elif target_c < c:
        return [3, 0, 0]  # Move left
    elif target_c > c:
        return [4, 0, 0]  # Move right
    return [0, 0, 0]  # No movement needed