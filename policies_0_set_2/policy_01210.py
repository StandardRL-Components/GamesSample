def policy(env):
    # Strategy: Maximize immediate reward by clearing largest connected groups (size>=4 for +5 bonus).
    # Prioritize groups under cursor to avoid movement cost, then target closest largest group.
    cx, cy = env.cursor_pos  # Current cursor position [x, y]
    
    # Check if current position has valid group (size>=2)
    if env.grid[cy, cx] != 0:
        current_group = env._find_connected_blocks(cx, cy)
        if len(current_group) >= 2:
            return [0, 1, 0]  # Clear group immediately
    
    # Find largest group (prioritize size>=4 for bonus)
    visited = set()
    best_size = 0
    best_pos = None
    for y in range(env.GRID_SIZE):
        for x in range(env.GRID_SIZE):
            if (y, x) in visited or env.grid[y, x] == 0:
                continue
            group = env._find_connected_blocks(x, y)
            size = len(group)
            if size < 2:
                continue
            visited.update(group)
            # Prefer larger groups, especially size>=4 for bonus
            if size > best_size or (size == best_size and size >= 4 and best_size < 4):
                best_size = size
                # Find closest cell in group to current cursor
                min_dist = float('inf')
                for gy, gx in group:
                    dist = abs(gx - cx) + abs(gy - cy)
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = (gx, gy)
    
    if best_pos is None:
        return [0, 0, 0]  # No valid groups
    
    # Move towards target group
    tx, ty = best_pos
    dx, dy = tx - cx, ty - cy
    if dx > 0:
        return [4, 0, 0]  # Right
    elif dx < 0:
        return [3, 0, 0]  # Left
    elif dy > 0:
        return [2, 0, 0]  # Down
    elif dy < 0:
        return [1, 0, 0]  # Up
    return [0, 0, 0]  # Shouldn't happen (already at target)