def policy(env):
    # Strategy: Minesweeper solver that uses internal state to deduce safe moves and flag mines.
    # Prioritizes revealing safe tiles first, then flags suspected mines. Uses cursor movement to explore.
    if env.game_over:
        return [0, 0, 0]
    
    visible_grid = env.visible_grid
    solution_grid = env.solution_grid
    cursor_r, cursor_c = env.cursor_pos
    
    # Deduce safe moves and mines from revealed numbers
    safe_moves = []
    mine_moves = []
    for r in range(env.GRID_ROWS):
        for c in range(env.GRID_COLS):
            if visible_grid[r, c] == 1 and solution_grid[r, c] > 0:
                hidden_neighbors = []
                flagged_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < env.GRID_ROWS and 0 <= nc < env.GRID_COLS:
                            if visible_grid[nr, nc] == 0:
                                hidden_neighbors.append((nr, nc))
                            elif visible_grid[nr, nc] == 2:
                                flagged_neighbors += 1
                if flagged_neighbors == solution_grid[r, c]:
                    safe_moves.extend(hidden_neighbors)
                elif flagged_neighbors + len(hidden_neighbors) == solution_grid[r, c]:
                    mine_moves.extend(hidden_neighbors)
    
    # Prioritize revealing safe tiles under cursor
    if (cursor_r, cursor_c) in safe_moves and visible_grid[cursor_r, cursor_c] == 0:
        return [0, 1, 0]
    
    # Flag mine under cursor if deduced
    if (cursor_r, cursor_c) in mine_moves and visible_grid[cursor_r, cursor_c] == 0:
        return [0, 0, 1]
    
    # Move to nearest safe or hidden tile
    target = None
    if safe_moves:
        target = min(safe_moves, key=lambda pos: abs(pos[0]-cursor_r) + abs(pos[1]-cursor_c))
    else:
        hidden_tiles = [(r, c) for r in range(env.GRID_ROWS) for c in range(env.GRID_COLS) 
                       if visible_grid[r, c] == 0 and (r, c) not in mine_moves]
        if hidden_tiles:
            target = min(hidden_tiles, key=lambda pos: abs(pos[0]-cursor_r) + abs(pos[1]-cursor_c))
    
    if target:
        tr, tc = target
        if cursor_r < tr:
            return [2, 0, 0]
        elif cursor_r > tr:
            return [1, 0, 0]
        elif cursor_c < tc:
            return [4, 0, 0]
        elif cursor_c > tc:
            return [3, 0, 0]
        return [0, 1, 0]
    
    return [0, 0, 0]