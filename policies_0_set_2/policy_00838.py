def policy(env):
    # Strategy: Prioritize swaps that create immediate matches (>=5 same color) for maximum reward.
    # If no match found, move cursor in a fixed pattern (right-down-left-up) to explore the board.
    def get_matched_cells(grid):
        matched = set()
        # Check horizontal matches
        for r in range(8):
            for c in range(8):
                if grid[r, c] == 0:
                    continue
                color = grid[r, c]
                count = 1
                for i in range(1, 8 - c):
                    if grid[r, c + i] == color:
                        count += 1
                    else:
                        break
                if count >= 5:
                    for i in range(count):
                        matched.add((r, c + i))
        # Check vertical matches
        for c in range(8):
            for r in range(8):
                if grid[r, c] == 0:
                    continue
                color = grid[r, c]
                count = 1
                for i in range(1, 8 - r):
                    if grid[r + i, c] == color:
                        count += 1
                    else:
                        break
                if count >= 5:
                    for i in range(count):
                        matched.add((r + i, c))
        return matched

    grid = env.grid
    x, y = env.cursor_pos
    best_dir = None
    best_count = 0

    for direction in [1, 2, 3, 4]:
        dx, dy = env.DIRECTIONS[direction]
        nx = (x + dx) % 8
        ny = (y + dy) % 8
        new_grid = grid.copy()
        new_grid[y, x], new_grid[ny, nx] = new_grid[ny, nx], new_grid[y, x]
        matches = get_matched_cells(new_grid)
        count = len(matches)
        if count > best_count:
            best_count = count
            best_dir = direction

    if best_count > 0:
        return [best_dir, 1, 0]

    for move_dir in [4, 2, 3, 1]:
        dx = 1 if move_dir == 4 else (-1 if move_dir == 3 else 0)
        dy = 1 if move_dir == 2 else (-1 if move_dir == 1 else 0)
        new_x = max(0, min(7, x + dx))
        new_y = max(0, min(7, y + dy))
        if (new_x, new_y) != (x, y):
            return [move_dir, 0, 0]

    return [0, 0, 0]