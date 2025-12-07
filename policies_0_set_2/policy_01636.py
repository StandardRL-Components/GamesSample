def policy(env):
    # Strategy: Maximize score by efficiently matching number pairs that sum to 10.
    # If a number is selected, move to its complement (10 - value). Otherwise, move to any number with an available match.
    # Press space when on target cell to select/match, minimizing movement and avoiding invalid actions.
    
    grid = env.grid
    selected_cell = env.selected_cell
    current_pos = env.cursor_pos
    GRID_COLS = env.GRID_COLS
    GRID_ROWS = env.GRID_ROWS
    
    def manhattan_dist(a, b):
        dx = min(abs(a[0]-b[0]), GRID_COLS - abs(a[0]-b[0]))
        dy = min(abs(a[1]-b[1]), GRID_ROWS - abs(a[1]-b[1]))
        return dx + dy

    if selected_cell is not None:
        sel_x, sel_y = selected_cell
        target_value = 10 - grid[sel_y][sel_x]
        candidates = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if grid[r][c] == target_value and (c, r) != selected_cell:
                    candidates.append((c, r))
        if not candidates:
            target_cell = selected_cell
        else:
            best_dist = float('inf')
            for cand in candidates:
                dist = manhattan_dist(current_pos, cand)
                if dist < best_dist:
                    best_dist = dist
                    target_cell = cand
    else:
        candidates = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if grid[r][c] is not None:
                    comp = 10 - grid[r][c]
                    for r2 in range(GRID_ROWS):
                        for c2 in range(GRID_COLS):
                            if (r2 != r or c2 != c) and grid[r2][c2] == comp:
                                candidates.append((c, r))
                                break
        if not candidates:
            target_cell = current_pos
        else:
            best_dist = float('inf')
            for cand in candidates:
                dist = manhattan_dist(current_pos, cand)
                if dist < best_dist:
                    best_dist = dist
                    target_cell = cand

    dx = target_cell[0] - current_pos[0]
    dy = target_cell[1] - current_pos[1]
    if dx > GRID_COLS//2:
        dx -= GRID_COLS
    elif dx < -GRID_COLS//2:
        dx += GRID_COLS
    if dy > GRID_ROWS//2:
        dy -= GRID_ROWS
    elif dy < -GRID_ROWS//2:
        dy += GRID_ROWS

    if dx == 0 and dy == 0:
        movement = 0
    else:
        if abs(dx) > abs(dy):
            movement = 4 if dx > 0 else 3
        else:
            movement = 2 if dy > 0 else 1

    space_action = 1 if (current_pos[0] == target_cell[0] and 
                         current_pos[1] == target_cell[1] and 
                         grid[current_pos[1]][current_pos[0]] is not None) else 0

    return [movement, space_action, 0]