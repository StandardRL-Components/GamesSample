def policy(env):
    # Prioritize harvesting ripe crops for immediate reward and future sales, then selling harvested crops for coins, then planting seeds on empty plots if affordable. Movement targets nearest high-value action to maximize efficiency.
    x, y = env.cursor_pos
    current_state = env.grid[y, x]
    
    # Harvest ripe crop if present
    if current_state == env.STATE_RIPE:
        return [0, 1, 0]
    
    # Sell harvested crops if available
    if env.harvested_crops > 0:
        return [0, 0, 1]
    
    # Find nearest ripe crop
    ripe_positions = []
    for i in range(env.GRID_SIZE):
        for j in range(env.GRID_SIZE):
            if env.grid[i, j] == env.STATE_RIPE:
                ripe_positions.append((j, i))
    if ripe_positions:
        min_dist = float('inf')
        nearest = None
        for tx, ty in ripe_positions:
            dist = abs(tx - x) + abs(ty - y)
            if dist < min_dist:
                min_dist = dist
                nearest = (tx, ty)
        dx, dy = nearest[0] - x, nearest[1] - y
        if abs(dx) > abs(dy):
            move = 3 if dx < 0 else 4
        else:
            move = 1 if dy < 0 else 2
        return [move, 0, 0]
    
    # Plant if affordable and on empty plot
    if env.score >= env.PLANT_COST and current_state == env.STATE_EMPTY:
        return [0, 1, 0]
    
    # Find nearest empty plot if affordable
    if env.score >= env.PLANT_COST:
        empty_positions = []
        for i in range(env.GRID_SIZE):
            for j in range(env.GRID_SIZE):
                if env.grid[i, j] == env.STATE_EMPTY:
                    empty_positions.append((j, i))
        if empty_positions:
            min_dist = float('inf')
            nearest = None
            for tx, ty in empty_positions:
                dist = abs(tx - x) + abs(ty - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (tx, ty)
            dx, dy = nearest[0] - x, nearest[1] - y
            if abs(dx) > abs(dy):
                move = 3 if dx < 0 else 4
            else:
                move = 1 if dy < 0 else 2
            return [move, 0, 0]
    
    # Default no-op if no actions available
    return [0, 0, 0]