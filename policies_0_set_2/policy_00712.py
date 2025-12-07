def policy(env):
    # Strategy: Prioritize firing at asteroids in current aim direction for immediate reward (100 score, +11 reward).
    # If no asteroid in current aim, change aim to direction with nearest asteroid within range to minimize wasted shots.
    # Upgrade only when no asteroids are in range and we have sufficient score (300) to increase projectile range for future efficiency.
    ship_x, ship_y = env.ship_pos
    current_dx, current_dy = env.aim_direction
    
    # Check if current aim has asteroid within range
    for step in range(1, env.projectile_range + 1):
        x = int(ship_x + current_dx * step)
        y = int(ship_y + current_dy * step)
        if not (0 <= x < env.GRID_WIDTH and 0 <= y < env.GRID_HEIGHT):
            break
        if (x, y) in env.asteroids:
            return [0, 1, 0]  # Fire with current aim
    
    # Find best direction with asteroid in range
    best_dir = 0
    min_dist = float('inf')
    for dir_val, (dx, dy) in enumerate([(0,-1), (0,1), (-1,0), (1,0)], 1):
        for step in range(1, env.projectile_range + 1):
            x = int(ship_x + dx * step)
            y = int(ship_y + dy * step)
            if not (0 <= x < env.GRID_WIDTH and 0 <= y < env.GRID_HEIGHT):
                break
            if (x, y) in env.asteroids:
                if step < min_dist:
                    min_dist = step
                    best_dir = dir_val
                break
    
    if best_dir != 0:
        return [best_dir, 0, 0]  # Change aim to best direction
    
    # Upgrade if no asteroids in range and conditions met
    if env.upgrade_level < 2 and env.score >= 300:
        return [0, 0, 1]  # Upgrade ship
    
    # Default: maintain current aim
    return [0, 0, 0]