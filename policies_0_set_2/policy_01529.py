def policy(env):
    # Strategy: Maximize immediate reward by breaking the block with highest potential chain reaction and row/column clears.
    # Prioritize blocks that yield the most points per move due to limited moves (30). Move cursor to best block if not already there.
    if env.game_over:
        return [0, 0, 0]
    
    rows, cols = env.grid.shape
    grid_list = env.grid.tolist()
    current_col, current_row = env.cursor_pos
    
    def simulate_break(grid, r, c):
        rows = len(grid)
        cols = len(grid[0])
        temp_grid = [row[:] for row in grid]
        if temp_grid[r][c] == 0:
            return 0
        grid_before = [row[:] for row in grid]
        
        queue = []
        queue.append((r, c))
        broken_set = set([(r, c)])
        temp_grid[r][c] = 0
        
        while queue:
            curr_r, curr_c = queue.pop(0)
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and temp_grid[nr][nc] > 0:
                    temp_grid[nr][nc] -= 1
                    if temp_grid[nr][nc] == 0 and (nr, nc) not in broken_set:
                        broken_set.add((nr, nc))
                        queue.append((nr, nc))
        
        num_broken = len(broken_set)
        rows_before_bool = [any(grid_before[i][j] > 0 for j in range(cols)) for i in range(rows)]
        rows_after_bool = [any(temp_grid[i][j] > 0 for j in range(cols)) for i in range(rows)]
        cleared_rows = sum(1 for i in range(rows) if rows_before_bool[i] and not rows_after_bool[i])
        cols_before_bool = [any(grid_before[i][j] > 0 for i in range(rows)) for j in range(cols)]
        cols_after_bool = [any(temp_grid[i][j] > 0 for i in range(rows)) for j in range(cols)]
        cleared_cols = sum(1 for j in range(cols) if cols_before_bool[j] and not cols_after_bool[j])
        
        return num_broken + 20 * cleared_rows + 10 * cleared_cols

    best_reward = -1
    best_cell = None
    for r in range(rows):
        for c in range(cols):
            if grid_list[r][c] > 0:
                reward = simulate_break(grid_list, r, c)
                if reward > best_reward:
                    best_reward = reward
                    best_cell = (c, r)
                    
    if best_cell is None:
        return [0, 0, 0]
        
    if (current_col, current_row) == best_cell:
        return [0, 1, 0]
        
    target_col, target_row = best_cell
    dx = target_col - current_col
    dy = target_row - current_row
    
    if abs(dx) > abs(dy):
        action0 = 4 if dx > 0 else 3
    else:
        action0 = 2 if dy > 0 else 1
        
    return [action0, 0, 0]