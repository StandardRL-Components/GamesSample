def policy(env):
    # Strategy: Prioritize targeting the lowest alien columns to prevent game loss, while maintaining optimal firing position. 
    # Always fire when possible to maximize destruction rate and row-clear bonuses. Break ties by horizontal proximity to minimize movement.
    if not env.aliens:
        return [0, 0, 0]
    
    # Find the lowest alien in each column to prioritize threats
    col_min_y = {}
    for alien in env.aliens:
        col = int((alien.centerx - 100) / 40)  # Approximate column from alien spawn logic
        if col not in col_min_y or alien.bottom > col_min_y[col][0]:
            col_min_y[col] = (alien.bottom, alien.centerx)
    
    if not col_min_y:
        return [0, 1, 0]
    
    # Find the most urgent target (lowest alien)
    max_bottom = max(y for y, _ in col_min_y.values())
    urgent_cols = [col for col in col_min_y if col_min_y[col][0] >= max_bottom - 5]
    
    # Choose closest urgent column to minimize movement
    target_x = min(col_min_y[col][1] for col in urgent_cols)
    dist = abs(env.player_pos.x - target_x)
    
    # Move toward target with threshold to avoid oscillation
    movement = 0
    if dist > 15:
        movement = 4 if env.player_pos.x < target_x else 3
    
    # Always fire when possible
    fire = 1 if env.player_fire_timer == 0 else 0
    
    return [movement, fire, 0]