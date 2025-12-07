def policy(env):
    # Strategy: Prioritize placing towers on path-adjacent cells for maximum coverage.
    # Use cheapest affordable tower to maximize quantity, favoring AOE for waves 3+.
    # Move cursor to best free cell based on path proximity and place when possible.
    current_cell = tuple(env.cursor_pos)
    current_tower = env.TOWER_TYPES[env.selected_tower_type_idx]
    current_cost = env.TOWER_SPECS[current_tower]["cost"]
    
    # Place tower if possible
    if current_cell not in env.occupied_cells and env.resources >= current_cost:
        return [0, 1, 0]
    
    # Cycle to optimal tower type
    desired_tower = "SINGLE"
    if env.wave_number >= 6:
        desired_tower = "SLOW"
    elif env.wave_number >= 3:
        desired_tower = "AOE"
    desired_idx = env.TOWER_TYPES.index(desired_tower)
    affordable = [t for t in env.TOWER_TYPES if env.TOWER_SPECS[t]["cost"] <= env.resources]
    if affordable and env.selected_tower_type_idx != desired_idx and env.resources >= env.TOWER_SPECS[desired_tower]["cost"]:
        return [0, 0, 1]
    if affordable and env.selected_tower_type_idx != env.TOWER_TYPES.index(min(affordable, key=lambda t: env.TOWER_SPECS[t]["cost"])):
        return [0, 0, 1]
    
    # Find best free cell (max path-adjacent coverage)
    best_score, best_cell = -1, None
    for x in range(env.grid_w):
        for y in range(env.grid_h):
            if (x, y) in env.occupied_cells:
                continue
            score = sum(1 for dx in (-1,0,1) for dy in (-1,0,1)
            if (x+dx, y+dy) in env.path_grid_coords)
            if score > best_score:
                best_score, best_cell = score, (x, y)
    
    # Move toward best cell
    if best_cell:
        cx, cy = env.cursor_pos
        tx, ty = best_cell
        if cx < tx: return [4, 0, 0]
        if cx > tx: return [3, 0, 0]
        if cy < ty: return [2, 0, 0]
        if cy > ty: return [1, 0, 0]
    
    return [0, 0, 0]