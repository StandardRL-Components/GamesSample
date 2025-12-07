def policy(env):
    # Strategy: Prioritize adjacent swaps that create matches for immediate reward. If none, move cursor in a raster scan to explore the board.
    if env.game_state != 'IDLE':
        return [0, 0, 0]

    def check_match(grid, x, y):
        gem_type = grid[y][x]
        if gem_type == -1:
            return False

        left = 0
        for i in range(x-1, -1, -1):
            if grid[y][i] == gem_type:
                left += 1
            else:
                break
        right = 0
        for i in range(x+1, env.GRID_WIDTH):
            if grid[y][i] == gem_type:
                right += 1
            else:
                break
        if left + right + 1 >= 3:
            return True

        up = 0
        for j in range(y-1, -1, -1):
            if grid[j][x] == gem_type:
                up += 1
            else:
                break
        down = 0
        for j in range(y+1, env.GRID_HEIGHT):
            if grid[j][x] == gem_type:
                down += 1
            else:
                break
        if up + down + 1 >= 3:
            return True

        return False

    cx, cy = env.cursor_pos
    grid = env.grid
    directions = [1, 2, 3, 4]
    dxdy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

    for d in directions:
        dx, dy = dxdy[d]
        tx, ty = cx + dx, cy + dy
        if 0 <= tx < env.GRID_WIDTH and 0 <= ty < env.GRID_HEIGHT:
            new_grid = [row[:] for row in grid]
            new_grid[cy][cx], new_grid[ty][tx] = new_grid[ty][tx], new_grid[cy][cx]
            if check_match(new_grid, cx, cy) or check_match(new_grid, tx, ty):
                return [d, 1, 0]

    next_x = (cx + 1) % env.GRID_WIDTH
    next_y = cy if next_x != 0 else (cy + 1) % env.GRID_HEIGHT
    dx = next_x - cx
    dy = next_y - cy

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