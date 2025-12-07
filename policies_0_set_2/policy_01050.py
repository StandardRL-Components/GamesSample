def policy(env):
    """
    Minesweeper policy that prioritizes safe reveals and flagging known mines based on number clues.
    Uses internal state (grid and revealed_grid) to deduce safe moves and avoid mines.
    Moves cursor to target cells with wrap-aware movement, then reveals or flags as appropriate.
    """
    GRID_SIZE = 9
    if env.game_over:
        return [0, 0, 0]
    
    cur_y, cur_x = env.cursor_pos
    grid = env.grid
    revealed = env.revealed_grid
    
    safe_cells = set()
    mine_cells = set()
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if revealed[y, x] == 1 and grid[y, x] > 0:
                number = grid[y, x]
                hidden_nbrs = []
                flagged_count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            if revealed[ny, nx] == 0:
                                hidden_nbrs.append((ny, nx))
                            elif revealed[ny, nx] == 2:
                                flagged_count += 1
                if flagged_count == number and hidden_nbrs:
                    safe_cells.update(hidden_nbrs)
                elif number - flagged_count == len(hidden_nbrs) and hidden_nbrs:
                    mine_cells.update(hidden_nbrs)
    
    safe_cells = {c for c in safe_cells if revealed[c[0]][c[1]] == 0}
    mine_cells = {c for c in mine_cells if revealed[c[0]][c[1]] == 0}
    
    def wrap_dist(a, b):
        return min(abs(a[0]-b[0]), GRID_SIZE-abs(a[0]-b[0])) + min(abs(a[1]-b[1]), GRID_SIZE-abs(a[1]-b[1]))
    
    if safe_cells:
        target = min(safe_cells, key=lambda c: wrap_dist(env.cursor_pos, c))
    elif mine_cells:
        target = min(mine_cells, key=lambda c: wrap_dist(env.cursor_pos, c))
    else:
        hidden = [(y, x) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if revealed[y, x] == 0]
        if not hidden:
            return [0, 0, 0]
        target = min(hidden, key=lambda c: wrap_dist(env.cursor_pos, c))
    
    ty, tx = target
    if (cur_y, cur_x) == (ty, tx):
        if target in safe_cells:
            return [0, 1, 0]
        elif target in mine_cells:
            return [0, 0, 1]
        else:
            return [0, 1, 0]
    
    dy = (ty - cur_y) % GRID_SIZE
    if dy > GRID_SIZE//2:
        dy -= GRID_SIZE
    dx = (tx - cur_x) % GRID_SIZE
    if dx > GRID_SIZE//2:
        dx -= GRID_SIZE
        
    if abs(dy) > abs(dx):
        move = 1 if dy < 0 else 2
    else:
        move = 3 if dx < 0 else 4
        
    return [move, 0, 0]