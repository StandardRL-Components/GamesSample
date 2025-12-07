def policy(env):
    """
    Maximizes reward by efficiently matching number pairs. Prioritizes moving to and selecting matching pairs when available, 
    otherwise moves towards the nearest valid number with a match. Avoids mismatches and unnecessary movements to conserve steps.
    """
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    grid = env.grid
    cursor = env.cursor_pos
    selected = env.selected_pos
    
    if selected is not None:
        sel_x, sel_y = selected
        num = grid[sel_y, sel_x]
        matches = []
        for y in range(env.GRID_SIZE):
            for x in range(env.GRID_SIZE):
                if grid[y, x] == num and (x, y) != selected:
                    matches.append((x, y))
        
        if matches:
            target = min(matches, key=lambda p: manhattan(cursor, p))
            dx = target[0] - cursor[0]
            dy = target[1] - cursor[1]
            if dx == 0 and dy == 0:
                return [0, 1, 0]
            if abs(dx) > abs(dy):
                return [4 if dx > 0 else 3, 0, 0]
            else:
                return [2 if dy > 0 else 1, 0, 0]
        else:
            return [0, 1, 0]
    
    else:
        candidates = []
        for y in range(env.GRID_SIZE):
            for x in range(env.GRID_SIZE):
                num = grid[y, x]
                if num != 0:
                    for y2 in range(env.GRID_SIZE):
                        for x2 in range(env.GRID_SIZE):
                            if (x2, y2) != (x, y) and grid[y2, x2] == num:
                                candidates.append((x, y))
                                break
                    else:
                        continue
                    break
        
        if not candidates:
            return [0, 0, 0]
        
        if (cursor[0], cursor[1]) in candidates:
            return [0, 1, 0]
        
        target = min(candidates, key=lambda p: manhattan(cursor, p))
        dx = target[0] - cursor[0]
        dy = target[1] - cursor[1]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]