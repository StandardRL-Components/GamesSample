def policy(env):
    # This policy greedily moves the currently selected color block in the direction that maximizes immediate matches with the target grid.
    # It prioritizes moves that increase the number of matching cells, avoiding invalid moves that hit obstacles or boundaries.
    # Since color changes don't consume moves, it focuses on optimal movement for the current selection without toggling colors.
    def simulate_shift(grid, direction, color_val):
        if direction == 1: dy, dx = -1, 0
        elif direction == 2: dy, dx = 1, 0
        elif direction == 3: dy, dx = 0, -1
        elif direction == 4: dy, dx = 0, 1
        else: return None
        moving_cells = []
        for i in range(12):
            for j in range(12):
                if grid[i, j] == color_val:
                    moving_cells.append((i, j))
        if not moving_cells:
            return None
        for (r, c) in moving_cells:
            nr, nc = r + dy, c + dx
            if not (0 <= nr < 12 and 0 <= nc < 12):
                return None
            dest_val = grid[nr, nc]
            if dest_val not in (0, color_val):
                return None
        new_grid = [list(row) for row in grid]
        for (r, c) in moving_cells:
            new_grid[r][c] = 0
        if dy == 1: moving_cells.sort(key=lambda x: x[0], reverse=True)
        elif dy == -1: moving_cells.sort(key=lambda x: x[0])
        elif dx == 1: moving_cells.sort(key=lambda x: x[1], reverse=True)
        elif dx == -1: moving_cells.sort(key=lambda x: x[1])
        for (r, c) in moving_cells:
            new_grid[r + dy][c + dx] = color_val
        return new_grid

    current_matches = 0
    for i in range(12):
        for j in range(12):
            if env.current_grid[i, j] == env.target_grid[i, j]:
                current_matches += 1
    best_delta = -1
    best_dir = 0
    color_val = env.MOVABLE_COLORS[env.selected_color_index]
    for direction in [1, 2, 3, 4]:
        new_grid = simulate_shift(env.current_grid, direction, color_val)
        if new_grid is None:
            continue
        new_matches = 0
        for i in range(12):
            for j in range(12):
                if new_grid[i][j] == env.target_grid[i, j]:
                    new_matches += 1
        delta = new_matches - current_matches
        if delta > best_delta:
            best_delta = delta
            best_dir = direction
    if best_delta > 0:
        return [best_dir, 0, 0]
    return [0, 0, 0]