def policy(env):
    # Strategy: Identify valid swaps that create matches by simulating potential moves.
    # If a valid swap is found, move the cursor to the required starting position and execute the swap.
    # If no swaps are available, reshuffle the board. This maximizes matches and minimizes wasted moves.
    if env.game_over or env._is_board_clear():
        return [0, 0, 0]
    
    # Check for valid swaps
    for r in range(env.GRID_ROWS):
        for c in range(env.GRID_COLS):
            # Check horizontal swap (right)
            if c < env.GRID_COLS - 1:
                temp_grid = env.grid.copy()
                temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                if env._find_matches_on_grid(temp_grid):
                    target_c = (c - 1) % env.GRID_COLS
                    if env.cursor_pos[0] == target_c and env.cursor_pos[1] == r:
                        return [4, 1, 0]
                    else:
                        current_c, current_r = env.cursor_pos
                        diff_c = (target_c - current_c) % env.GRID_COLS
                        if diff_c > env.GRID_COLS // 2:
                            diff_c -= env.GRID_COLS
                        diff_r = (r - current_r) % env.GRID_ROWS
                        if diff_r > env.GRID_ROWS // 2:
                            diff_r -= env.GRID_ROWS
                        if abs(diff_r) > abs(diff_c):
                            return [2 if diff_r > 0 else 1, 0, 0]
                        else:
                            return [4 if diff_c > 0 else 3, 0, 0]
            # Check vertical swap (down)
            if r < env.GRID_ROWS - 1:
                temp_grid = env.grid.copy()
                temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                if env._find_matches_on_grid(temp_grid):
                    target_r = (r - 1) % env.GRID_ROWS
                    if env.cursor_pos[0] == c and env.cursor_pos[1] == target_r:
                        return [2, 1, 0]
                    else:
                        current_c, current_r = env.cursor_pos
                        diff_c = (c - current_c) % env.GRID_COLS
                        if diff_c > env.GRID_COLS // 2:
                            diff_c -= env.GRID_COLS
                        diff_r = (target_r - current_r) % env.GRID_ROWS
                        if diff_r > env.GRID_ROWS // 2:
                            diff_r -= env.GRID_ROWS
                        if abs(diff_r) > abs(diff_c):
                            return [2 if diff_r > 0 else 1, 0, 0]
                        else:
                            return [4 if diff_c > 0 else 3, 0, 0]
    return [0, 0, 1]