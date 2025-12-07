def policy(env):
    # Strategy: Place towers at optimal path choke points to maximize zombie coverage and damage.
    # Prioritize Frost towers for slowing, then Cannons for high damage, then Gatlings for cleanup.
    # Move cursor to best available tile and place if affordable, else cycle to cheaper tower type.
    
    # Predefined optimal tower positions near path bends for maximum coverage
    candidates = [
        (2, 4), (4, 4), (3, 3), (3, 5),  # Around first bend (3,4)
        (2, 1), (4, 1), (3, 0), (3, 2),   # Around second bend (3,1)
        (7, 1), (9, 1), (8, 0), (8, 2),   # Around third bend (8,1)
        (7, 8), (9, 8), (8, 7), (8, 9),   # Around fourth bend (8,8)
        (11, 8), (13, 8), (12, 7), (12, 9), # Around fifth bend (12,8)
        (11, 4), (13, 4), (12, 3), (12, 5)  # Around final bend (12,4)
    ]
    
    # Check if current tower type is affordable
    current_cost = env.TOWER_SPECS[env.selected_tower_type]['cost']
    if env.resources < current_cost:
        return [0, 0, 1]  # Cycle to cheaper tower
    
    # Find best available candidate position
    target = None
    for cand in candidates:
        if cand in env.path:
            continue
        if any(t['grid_pos'] == list(cand) for t in env.towers):
            continue
        target = cand
        break
    
    if target is None:
        return [0, 0, 0]  # No valid positions
    
    # Move toward target position
    cx, cy = env.cursor_pos
    tx, ty = target
    if cx < tx:
        return [4, 0, 0]  # Move right
    elif cx > tx:
        return [3, 0, 0]  # Move left
    elif cy < ty:
        return [2, 0, 0]  # Move down
    elif cy > ty:
        return [1, 0, 0]  # Move up
    else:
        return [0, 1, 0]  # Place tower