def policy(env):
    # Strategy: Prioritize blocking enemy paths to protect fortress health, then build inner defensive wall.
    # Use strongest (Steel) blocks for durability. Move efficiently to target cells and place blocks when aligned.
    inner_wall = [(2,2), (3,2), (4,2), (5,2), (6,2), (7,2),
                 (2,7), (3,7), (4,7), (5,7), (6,7), (7,7),
                 (2,3), (2,4), (2,5), (2,6),
                 (7,3), (7,4), (7,5), (7,6)]
    
    if env.selected_block_type_idx != 2:
        return [0, 0, 1]
    
    closest_enemy = None
    min_dist = float('inf')
    for enemy in env.enemies:
        gx, gy = enemy.get_grid_pos()
        if 0 <= gx < env.GRID_SIZE and 0 <= gy < env.GRID_SIZE:
            dist = abs(gx-5) + abs(gy-5)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
                
    if min_dist <= 3 and closest_enemy is not None:
        gx, gy = closest_enemy.get_grid_pos()
        dx = 5 - gx
        dy = 5 - gy
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        target_cell = (gx + step_x, gy + step_y)
        if (0 <= target_cell[0] < env.GRID_SIZE and 0 <= target_cell[1] < env.GRID_SIZE and
            target_cell not in env.grid):
            cx, cy = env.cursor_pos
            if cx == target_cell[0] and cy == target_cell[1]:
                return [0, 1, 0]
            dx_move = target_cell[0] - cx
            dy_move = target_cell[1] - cy
            if dx_move != 0:
                return [4 if dx_move > 0 else 3, 0, 0]
            else:
                return [2 if dy_move > 0 else 1, 0, 0]
                
    empty_inner = [cell for cell in inner_wall if cell not in env.grid]
    if empty_inner:
        cx, cy = env.cursor_pos
        target_cell = min(empty_inner, key=lambda c: abs(c[0]-cx) + abs(c[1]-cy))
        if cx == target_cell[0] and cy == target_cell[1]:
            return [0, 1, 0]
        dx_move = target_cell[0] - cx
        dy_move = target_cell[1] - cy
        if dx_move != 0:
            return [4 if dx_move > 0 else 3, 0, 0]
        else:
            return [2 if dy_move > 0 else 1, 0, 0]
            
    return [0, 0, 0]