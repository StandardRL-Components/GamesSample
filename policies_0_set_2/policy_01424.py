def policy(env):
    # Strategy: Create a serpentine maze by alternating block rows per column, starting from the right near the base.
    # This maximizes path length for enemies, delaying them until wave completion while minimizing base damage.
    target = None
    for col in range(17, -1, -1):
        row_range = range(1, 11) if col % 2 == 1 else range(0, 10)
        for row in row_range:
            if env.grid[col, row] == env.BLOCK_EMPTY and (col, row) not in env.base_cells:
                target = (col, row)
                break
        if target is not None:
            break
    if target is None:
        return [0, 0, 0]
    
    cx, cy = env.cursor_pos
    tx, ty = target
    if cx == tx and cy == ty:
        a2 = 1 if env.wave_number >= 3 else 0
        return [0, 1, a2]
    
    dx = tx - cx
    dy = ty - cy
    if dx > env.GRID_COLS / 2:
        dx -= env.GRID_COLS
    elif dx < -env.GRID_COLS / 2:
        dx += env.GRID_COLS
    if dy > env.GRID_ROWS / 2:
        dy -= env.GRID_ROWS
    elif dy < -env.GRID_ROWS / 2:
        dy += env.GRID_ROWS
        
    if abs(dx) > abs(dy):
        a0 = 4 if dx > 0 else 3
    else:
        a0 = 2 if dy > 0 else 1
    return [a0, 0, 0]