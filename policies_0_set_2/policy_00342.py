def policy(env):
    # Strategy: Prioritize placing crystals adjacent to unlit targets in the light path to maximize immediate rewards.
    # Move towards nearest unlit target, place crystal if adjacent and in light path, else cycle crystal type if unavailable.
    cursor_x, cursor_y = env.cursor_pos
    grid = env.grid
    crystals_left = env.crystal_inventory[env.selected_crystal_type]
    
    # Find unlit targets
    unlit_targets = [t for t in env.targets if not t['lit']]
    if not unlit_targets:
        return [0, 0, 0]  # All targets lit, do nothing
    
    # Get nearest unlit target
    target_pos = unlit_targets[0]['pos']
    dx = target_pos[0] - cursor_x
    dy = target_pos[1] - cursor_y
    
    # If adjacent to target and empty, place crystal if available
    if abs(dx) + abs(dy) == 1 and grid[cursor_y][cursor_x] == 0:
        if crystals_left > 0:
            return [0, 1, 0]
        else:
            return [0, 0, 1]  # Cycle crystal type if none left
    
    # Move towards target
    if dx != 0:
        move_x = 4 if dx > 0 else 3
        if 0 <= cursor_x + (1 if dx > 0 else -1) < env.GRID_WIDTH and grid[cursor_y][cursor_x + (1 if dx > 0 else -1)] == 0:
            return [move_x, 0, 0]
    if dy != 0:
        move_y = 2 if dy > 0 else 1
        if 0 <= cursor_y + (1 if dy > 0 else -1) < env.GRID_HEIGHT and grid[cursor_y + (1 if dy > 0 else -1)][cursor_x] == 0:
            return [move_y, 0, 0]
    
    # Default: no movement
    return [0, 0, 0]