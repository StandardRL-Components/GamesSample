def policy(env):
    # This policy maximizes reward by systematically finding and executing valid gem swaps that create matches.
    # It scans the grid for the first valid swap (prioritizing right then bottom neighbors) and moves the cursor to perform the swap.
    # If a tile is already selected, it moves to the adjacent swap partner and completes the swap.
    # Movement uses wrapped distance to handle grid edges efficiently.
    if env.game_state != 'IDLE':
        return [0, 0, 0]
    
    grid = env.grid
    cursor_pos = env.cursor_pos
    selected_tile = env.selected_tile
    
    def find_valid_swap():
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH):
                if c < env.GRID_WIDTH - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if env._find_matches(temp_grid):
                        return (r, c), (r, c+1)
                if r < env.GRID_HEIGHT - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if env._find_matches(temp_grid):
                        return (r, c), (r+1, c)
        return None
    
    if selected_tile is None:
        swap = find_valid_swap()
        if swap is None:
            return [0, 0, 0]
        tile1, tile2 = swap
        target = (tile1[1], tile1[0])
        if cursor_pos == target:
            return [0, 1, 0]
        dx = (target[0] - cursor_pos[0]) % env.GRID_WIDTH
        dy = (target[1] - cursor_pos[1]) % env.GRID_HEIGHT
        if dx > env.GRID_WIDTH // 2:
            dx -= env.GRID_WIDTH
        if dy > env.GRID_HEIGHT // 2:
            dy -= env.GRID_HEIGHT
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    else:
        swap = None
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH):
                if c < env.GRID_WIDTH - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if env._find_matches(temp_grid) and ((r, c) == selected_tile or (r, c+1) == selected_tile):
                        swap = ((r, c), (r, c+1))
                        break
                if r < env.GRID_HEIGHT - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if env._find_matches(temp_grid) and ((r, c) == selected_tile or (r+1, c) == selected_tile):
                        swap = ((r, c), (r+1, c))
                        break
            if swap is not None:
                break
        if swap is None:
            return [0, 0, 1]
        tile1, tile2 = swap
        other_tile = tile2 if tile1 == selected_tile else tile1
        target = (other_tile[1], other_tile[0])
        if cursor_pos == target:
            return [0, 1, 0]
        dx = (target[0] - cursor_pos[0]) % env.GRID_WIDTH
        dy = (target[1] - cursor_pos[1]) % env.GRID_HEIGHT
        if dx > env.GRID_WIDTH // 2:
            dx -= env.GRID_WIDTH
        if dy > env.GRID_HEIGHT // 2:
            dy -= env.GRID_HEIGHT
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]