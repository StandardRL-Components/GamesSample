def policy(env):
    # Strategy: Systematically build a defensive maze near the fortress to maximize path length for enemies.
    # Prioritize blocking enemies close to fortress, then fill columns from right to left to create optimal barriers.
    current_x, current_y = env.cursor_pos
    closest_enemy = None
    min_dist = float('inf')
    for enemy in env.enemies:
        ex, ey = enemy['pos']
        dist = min(abs(ex - fx) + abs(ey - fy) for fx, fy in env.fortress_cells)
        if dist < min_dist:
            min_dist = dist
            closest_enemy = enemy

    if min_dist < 5 and closest_enemy and closest_enemy['path'] and len(closest_enemy['path']) > 1:
        next_x, next_y = closest_enemy['path'][1]
        desired_x, desired_y = next_x, next_y
    else:
        col_offset = (env.steps // env.GRID_HEIGHT) % (env.GRID_WIDTH - 2)
        desired_x = env.GRID_WIDTH - 3 - col_offset
        desired_y = env.steps % env.GRID_HEIGHT

    if (desired_x, desired_y) in env.fortress_cells:
        col_offset = (env.steps // env.GRID_HEIGHT) % (env.GRID_WIDTH - 2)
        desired_x = env.GRID_WIDTH - 3 - col_offset
        desired_y = env.steps % env.GRID_HEIGHT

    movement = 0
    if current_x < desired_x:
        movement = 4
    elif current_x > desired_x:
        movement = 3
    else:
        if current_y < desired_y:
            movement = 2
        elif current_y > desired_y:
            movement = 1

    return [movement, 1, 0]