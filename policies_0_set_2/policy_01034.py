def policy(env):
    # Strategy: Maximize alignment by moving crystals to positions with the most adjacent crystals (horizontal/vertical).
    # Prioritizes creating and extending chains to reach the goal of 5 in a row/column efficiently.
    if env.game_over:
        return [0, 0, 0]
    
    cur_x, cur_y = env.cursor_pos
    empty_cells = []
    for x in range(env.GRID_WIDTH):
        for y in range(env.GRID_HEIGHT):
            if env.walls[x][y] == 0 and (x, y) not in env.crystals:
                empty_cells.append((x, y))
    
    if not empty_cells:
        return [0, 0, 0]
    
    best_score = -1
    best_cell = empty_cells[0]
    for cell in empty_cells:
        x, y = cell
        score = 0
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT:
                if (nx, ny) in env.crystals:
                    score += 1
        if score > best_score:
            best_score = score
            best_cell = cell
        elif score == best_score:
            if cell[1] < best_cell[1] or (cell[1] == best_cell[1] and cell[0] < best_cell[0]):
                best_cell = cell
    
    if (cur_x, cur_y) == best_cell:
        return [0, 1, 0]
    else:
        dx = best_cell[0] - cur_x
        dy = best_cell[1] - cur_y
        if dx > 0:
            return [4, 0, 0]
        elif dx < 0:
            return [3, 0, 0]
        elif dy > 0:
            return [2, 0, 0]
        elif dy < 0:
            return [1, 0, 0]
        else:
            return [0, 0, 0]