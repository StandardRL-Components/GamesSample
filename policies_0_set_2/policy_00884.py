def policy(env):
    # Strategy: Keep rider moving right by drawing safe track segments ahead. Prioritize drawing when rider is airborne or track is ending to prevent crashes and maintain momentum.
    rider_x, rider_y = env.rider_pos.x, env.rider_pos.y
    cursor_x, cursor_y = env.cursor_pos.x, env.cursor_pos.y
    target_x = rider_x + 100  # Look ahead distance
    
    # Find terrain height at target_x
    terrain_y = None
    for i in range(len(env.terrain_points)-1):
        x1, y1 = env.terrain_points[i]
        x2, y2 = env.terrain_points[i+1]
        if x1 <= target_x <= x2:
            terrain_y = y1 + (y2 - y1) * (target_x - x1) / (x2 - x1)
            break
    if terrain_y is None:
        terrain_y = env.terrain_points[-1][1]
    
    target_y = max(0, min(terrain_y - 30, rider_y + 20))  # Safe height above terrain
    
    dx = target_x - cursor_x
    dy = target_y - cursor_y
    
    movement = 0
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1 if dy < 0 else 0
    
    last_track_x = env.track_points[-1].x
    space_held = 1 if (env.is_airborne or last_track_x - rider_x < 50) and abs(dx) < 20 and abs(dy) < 20 else 0
    
    return [movement, space_held, 0]