def policy(env):
    # Strategy: Prioritize collecting fuel until reaching 100, then head to escape zone.
    # Avoid aliens by moving away when too close, using repulsion vectors.
    # Combine attraction to target (fuel/exit) and repulsion from aliens to choose best movement direction.
    player_x, player_y = env.player['x'], env.player['y']
    
    # Determine target: exit if fuel >= 100, else nearest fuel cell
    if env.fuel >= env.WIN_FUEL:
        target_x = env.escape_zone.x + env.escape_zone.width / 2
        target_y = env.escape_zone.y + env.escape_zone.height / 2
    else:
        min_dist = float('inf')
        target_x, target_y = player_x, player_y
        for cell in env.fuel_cells:
            cx = cell.x + cell.width / 2
            cy = cell.y + cell.height / 2
            dist = ((player_x - cx) ** 2 + (player_y - cy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist, target_x, target_y = dist, cx, cy
    
    # Calculate attraction vector to target
    dx_target = target_x - player_x
    dy_target = target_y - player_y
    dist_target = (dx_target**2 + dy_target**2) ** 0.5
    if dist_target > 0:
        dx_target /= dist_target
        dy_target /= dist_target
    
    # Calculate repulsion from nearby aliens
    dx_repel, dy_repel = 0, 0
    danger_radius = 100
    for alien in env.aliens:
        dx = player_x - alien['x']
        dy = player_y - alien['y']
        dist = (dx**2 + dy**2) ** 0.5
        if dist < danger_radius:
            weight = (danger_radius - dist) / danger_radius
            dx_repel += (dx / dist) * weight
            dy_repel += (dy / dist) * weight
    
    # Combine vectors
    dx_total = dx_target + dx_repel * 2  # Repulsion has higher weight
    dy_total = dy_target + dy_repel * 2
    
    # Score each movement direction
    scores = [0, -dy_total, dy_total, -dx_total, dx_total]  # [none, up, down, left, right]
    best_action = 0
    for i in range(1, 5):
        if scores[i] > scores[best_action]:
            best_action = i
    
    return [best_action, 0, 0]