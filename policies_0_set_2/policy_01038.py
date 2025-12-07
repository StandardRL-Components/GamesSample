def policy(env):
    # Strategy: Maximize immediate reward by always clicking on the empty cell with the highest potential fill (self + empty neighbors). 
    # Move directly toward the best candidate cell to minimize wasted moves, ensuring optimal use of limited moves (15).
    if env.game_over:
        return [0, 0, 0]
    
    cursor_x, cursor_y = env.cursor_pos
    best_score = -1
    best_cell = None
    for y in range(env.GRID_SIZE):
        for x in range(env.GRID_SIZE):
            if env.grid[y, x] == 0:
                score = 1
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < env.GRID_SIZE and 0 <= ny < env.GRID_SIZE and env.grid[ny, nx] == 0:
                        score += 1
                if score > best_score or (score == best_score and best_cell is None):
                    best_score = score
                    best_cell = (x, y)
    
    if best_cell is None:
        return [0, 0, 0]
    
    t_x, t_y = best_cell
    if cursor_x == t_x and cursor_y == t_y:
        return [0, 1, 0]
    
    dx = t_x - cursor_x
    dy = t_y - cursor_y
    if dx != 0:
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dy > 0 else 1, 0, 0]