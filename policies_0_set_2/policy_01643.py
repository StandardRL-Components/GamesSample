def policy(env):
    """
    Strategy: Prioritize selling harvested crops for immediate gold, then harvest ready crops for 
    reward and to free plots, then plant seeds on empty plots to maximize future harvests. 
    Movement targets nearest ready crop or empty plot (if gold allows), using consistent tie-breaking.
    """
    # Sell if we have harvested crops to get gold for planting
    if env.harvested_crops > 0:
        return [0, 0, 1]
    
    r, c = env.selector_pos
    current_plot = env.farm_grid[r, c]
    
    # Harvest if current plot is ready
    if current_plot == env.CROP_STATE_READY:
        return [0, 1, 0]
    
    # Plant if current plot is empty and we have enough gold
    if current_plot == env.CROP_STATE_EMPTY and env.gold >= env.PLANT_COST:
        return [0, 1, 0]
    
    # Find best movement target: ready crops first, then empty plots if affordable
    targets = []
    grid_size = env.GRID_SIZE
    for i in range(grid_size):
        for j in range(grid_size):
            state = env.farm_grid[i, j]
            if state == env.CROP_STATE_READY:
                targets.append((i, j, 0))  # Priority 0: ready crops
            elif state == env.CROP_STATE_EMPTY and env.gold >= env.PLANT_COST:
                targets.append((i, j, 1))  # Priority 1: empty plots
    
    if not targets:
        return [0, 0, 0]  # No valid targets, wait
    
    # Find closest target using Manhattan distance with wrap-around
    best_dist = float('inf')
    best_target = None
    for i, j, priority in targets:
        # Calculate wrapped distance
        dr = min(abs(i - r), grid_size - abs(i - r))
        dc = min(abs(j - c), grid_size - abs(j - c))
        dist = dr + dc
        # Prefer ready crops (priority 0) over empty plots at same distance
        if dist < best_dist or (dist == best_dist and priority < best_target[2]):
            best_dist = dist
            best_target = (i, j, priority)
    
    ti, tj, _ = best_target
    # Calculate direction considering grid wrap
    dr = (ti - r) % grid_size
    if dr > grid_size // 2:
        dr -= grid_size
    dc = (tj - c) % grid_size
    if dc > grid_size // 2:
        dc -= grid_size
    
    # Move in direction of largest displacement
    if abs(dr) > abs(dc):
        return [1 if dr < 0 else 2, 0, 0]  # Up or down
    elif abs(dc) > 0:
        return [3 if dc < 0 else 4, 0, 0]  # Left or right
    return [0, 0, 0]  # Already at target