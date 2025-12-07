def policy(env):
    # Strategy: Maximize coverage of enemy path by building towers at optimal positions.
    # Prioritize Cannon towers early for cost-efficiency, Missile later for damage.
    # Move cursor to the best available tile (max path coverage) and build if affordable.
    # Cycle tower type only when necessary to match desired type based on wave and gold.
    
    # Read current state
    gold = env.gold
    wave = env.wave_number
    current_idx = env.selected_tower_type_idx
    current_type = env.tower_types[current_idx]
    cannon_cost = env.TOWER_SPECS["Cannon"]["cost"]
    missile_cost = env.TOWER_SPECS["Missile"]["cost"]
    occupied = set(t["pos"] for t in env.towers)
    
    # Determine desired tower type
    if wave < 5:
        desired_type = "Cannon" if gold >= cannon_cost else None
    else:
        if gold >= missile_cost:
            desired_type = "Missile"
        elif gold >= cannon_cost:
            desired_type = "Cannon"
        else:
            desired_type = None
    
    # Cycle tower type if needed
    a2 = 0
    if desired_type is not None and env.tower_types.index(desired_type) != current_idx:
        a2 = 1
    
    # If cannot build, return no-op
    if desired_type is None:
        return [0, 0, a2]
    
    # Find best buildable tile (max path coverage)
    best_tile = None
    best_score = -1
    range_px = env.TOWER_SPECS[desired_type]["range"] * env.TILE_WIDTH_HALF * 1.5
    sq_range = range_px ** 2
    for tile in env.buildable_tiles:
        if tile in occupied:
            continue
        tile_x, tile_y = tile
        iso_x = env.ISO_OFFSET_X + (tile_x - tile_y) * env.TILE_WIDTH_HALF
        iso_y = env.ISO_OFFSET_Y + (tile_x + tile_y) * env.TILE_HEIGHT_HALF
        score = 0
        for path_tile in env.path_grid_coords:
            path_iso_x = env.ISO_OFFSET_X + (path_tile[0] - path_tile[1]) * env.TILE_WIDTH_HALF
            path_iso_y = env.ISO_OFFSET_Y + (path_tile[0] + path_tile[1]) * env.TILE_HEIGHT_HALF
            dx = iso_x - path_iso_x
            dy = iso_y - path_iso_y
            if dx*dx + dy*dy <= sq_range:
                score += 1
        if score > best_score or (score == best_score and tile < best_tile):
            best_score = score
            best_tile = tile

    # If no available tile, return no-op
    if best_tile is None:
        return [0, 0, a2]
    
    # Move cursor toward best tile
    dx = best_tile[0] - env.cursor_pos[0]
    dy = best_tile[1] - env.cursor_pos[1]
    a0 = 0
    if dx != 0:
        a0 = 4 if dx > 0 else 3
    elif dy != 0:
        a0 = 2 if dy > 0 else 1
        
    # Build if on best tile and affordable
    a1 = 0
    cost = env.TOWER_SPECS[desired_type]["cost"]
    if env.cursor_pos[0] == best_tile[0] and env.cursor_pos[1] == best_tile[1] and gold >= cost:
        a1 = 1
        
    return [a0, a1, a2]