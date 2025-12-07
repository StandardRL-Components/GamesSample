def policy(env):
    # Strategy: Prioritize placing cannons (type 0) at spots with maximum path coverage to maximize enemy engagements and kill rewards. Systematically scan grid for optimal placements, moving efficiently between positions. Always ensure cannon is selected and only place when affordable.
    if env.selected_tower_type != 0:
        return [0, 0, 1]  # Switch to cannon (type 0)
    
    cannon_cost = env.TOWER_TYPES[0]['cost']
    if env.gold < cannon_cost:
        return [0, 0, 0]  # Wait until affordable
    
    best_spot = None
    best_coverage = -1
    range_sq = env.TOWER_TYPES[0]['range'] ** 2
    
    for idx, spot in enumerate(env.placement_spots):
        if spot['occupied']:
            continue
        coverage = 0
        for path_point in env.path:
            dx = spot['pos'][0] - path_point[0]
            dy = spot['pos'][1] - path_point[1]
            if dx*dx + dy*dy <= range_sq:
                coverage += 1
        if coverage > best_coverage:
            best_coverage = coverage
            best_spot = idx
    
    if best_spot is None:
        return [0, 0, 0]  # No available spots
    
    current_idx = env.cursor_index
    if current_idx == best_spot:
        return [0, 1, 0]  # Place tower at optimal spot
    
    n_cols = 5
    current_row, current_col = current_idx // n_cols, current_idx % n_cols
    target_row, target_col = best_spot // n_cols, best_spot % n_cols
    
    if current_col < target_col:
        return [4, 0, 0]  # Move right
    elif current_col > target_col:
        return [3, 0, 0]  # Move left
    elif current_row < target_row:
        return [2, 0, 0]  # Move down
    elif current_row > target_row:
        return [1, 0, 0]  # Move up
    return [0, 0, 0]