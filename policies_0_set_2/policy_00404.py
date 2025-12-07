def policy(env):
    # Strategy: Prioritize harvesting mature crops for immediate reward, then plant seeds to maximize future yields.
    # Only sell when inventory is substantial to minimize trips. Use efficient pathing to reduce time waste.
    cx, cy = env.cursor_pos
    mx, my = env.market_pos
    total_crops = sum(env.harvested_crops.values())
    
    # Sell if at market with sufficient inventory (minimizes trips)
    if cx == mx and cy == my and total_crops >= 15:
        return [0, 0, 1]
    
    # Harvest mature crop if present
    plot = env.farm_grid[cy][cx]
    if plot["state"] == "planted":
        crop_info = env.CROP_TYPES[plot["type"]]
        if plot["growth"] >= crop_info["growth_time"]:
            return [0, 0, 1]
    
    # Move to market if inventory is large enough
    if total_crops >= 15:
        if cx < mx: return [4, 0, 0]
        if cx > mx: return [3, 0, 0]
        if cy < my: return [2, 0, 0]
        if cy > my: return [1, 0, 0]
    
    # Find nearest mature crop
    best_dist = float('inf')
    target = None
    for y in range(env.GRID_ROWS):
        for x in range(env.GRID_COLS):
            plot = env.farm_grid[y][x]
            if plot["state"] == "planted":
                crop_info = env.CROP_TYPES[plot["type"]]
                if plot["growth"] >= crop_info["growth_time"]:
                    dist = abs(x - cx) + abs(y - cy)
                    if dist < best_dist:
                        best_dist = dist
                        target = (x, y)
    if target:
        tx, ty = target
        if cx < tx: return [4, 0, 0]
        if cx > tx: return [3, 0, 0]
        if cy < ty: return [2, 0, 0]
        if cy > ty: return [1, 0, 0]
    
    # Plant in current plot if empty
    if env.farm_grid[cy][cx]["state"] == "empty":
        return [0, 1, 0]
    
    # Find nearest empty plot
    best_dist = float('inf')
    target = None
    for y in range(env.GRID_ROWS):
        for x in range(env.GRID_COLS):
            if env.farm_grid[y][x]["state"] == "empty":
                dist = abs(x - cx) + abs(y - cy)
                if dist < best_dist:
                    best_dist = dist
                    target = (x, y)
    if target:
        tx, ty = target
        if cx < tx: return [4, 0, 0]
        if cx > tx: return [3, 0, 0]
        if cy < ty: return [2, 0, 0]
        if cy > ty: return [1, 0, 0]
    
    return [0, 0, 0]