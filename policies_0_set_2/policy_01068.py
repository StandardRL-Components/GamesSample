def policy(env):
    # Strategy: Prioritize collecting immediate matches under cursor, then move to nearest valid match group.
    # Break ties by targeting larger groups first, then proximity. Avoid no-ops by checking match validity.
    rows, cols = env.GRID_SIZE[1], env.GRID_SIZE[0]
    cx, cy = env.cursor_pos
    grid = env.grid
    
    # Check current position for valid match
    if grid[cy, cx] != -1:
        matches = env._find_matches(cx, cy)
        if len(matches) >= 3:
            return [0, 1, 0]
    
    # Find all valid match groups
    groups = []
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited or grid[r, c] == -1:
                continue
            matches = env._find_matches(c, r)
            if len(matches) >= 3:
                groups.append(matches)
                visited.update(matches)
    
    if not groups:
        return [0, 0, 0]
    
    # Find best target group (largest then closest)
    best_group = max(groups, key=lambda g: (len(g), -min(
        min(abs(r - cy), rows - abs(r - cy)) + min(abs(c - cx), cols - abs(c - cx))
        for (r, c) in g
    )))
    
    # Find closest cell in best group
    target_r, target_c = min(best_group, key=lambda rc: (
        min(abs(rc[0] - cy), rows - abs(rc[0] - cy)) + 
        min(abs(rc[1] - cx), cols - abs(rc[1] - cx))
    ))
    
    # Calculate wrapped movement directions
    dx = (target_c - cx) % cols
    if dx > cols // 2:
        dx -= cols
    dy = (target_r - cy) % rows
    if dy > rows // 2:
        dy -= rows
    
    # Move in primary direction first
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    elif dy != 0:
        return [2 if dy > 0 else 1, 0, 0]
    return [0, 1, 0]