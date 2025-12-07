def policy(env):
    # Strategy: Prioritize immediate matches by evaluating all possible swaps from the current cursor position.
    # If no match is found, cycle swap direction to check other neighbors. If still no match, move cursor to
    # the nearest cell with a valid swap opportunity to maximize score and minimize penalties.
    def has_match(grid):
        for r in range(env.GRID_SIZE):
            for c in range(env.GRID_SIZE):
                gem = grid[r, c]
                if gem == -1:
                    continue
                if c < env.GRID_SIZE - 2 and grid[r, c+1] == gem and grid[r, c+2] == gem:
                    return True
                if r < env.GRID_SIZE - 2 and grid[r+1, c] == gem and grid[r+2, c] == gem:
                    return True
        return False

    r, c = env.cursor_pos
    current_dir = env.swap_direction_idx
    directions = [0, 1, 2, 3]
    for d in [current_dir] + [i for i in directions if i != current_dir]:
        if d == 0:
            r2 = (r - 1) % env.GRID_SIZE
            c2 = c
        elif d == 1:
            r2 = r
            c2 = (c + 1) % env.GRID_SIZE
        elif d == 2:
            r2 = (r + 1) % env.GRID_SIZE
            c2 = c
        else:
            r2 = r
            c2 = (c - 1) % env.GRID_SIZE
        grid_copy = env.grid.copy()
        grid_copy[r, c], grid_copy[r2, c2] = grid_copy[r2, c2], grid_copy[r, c]
        if has_match(grid_copy):
            if d == current_dir:
                return [0, 1, 0]
            else:
                return [0, 0, 1]

    valid_cells = set()
    for rr in range(env.GRID_SIZE):
        for cc in range(env.GRID_SIZE):
            for dx, dy in [(0,1), (1,0)]:
                rr2 = (rr + dx) % env.GRID_SIZE
                cc2 = (cc + dy) % env.GRID_SIZE
                grid_copy = env.grid.copy()
                grid_copy[rr, cc], grid_copy[rr2, cc2] = grid_copy[rr2, cc2], grid_copy[rr, cc]
                if has_match(grid_copy):
                    valid_cells.add((rr, cc))
                    valid_cells.add((rr2, cc2))
    if not valid_cells:
        return [0, 0, 0]
    best_dist = float('inf')
    move = 0
    for cell in valid_cells:
        dr = min(abs(cell[0] - r), env.GRID_SIZE - abs(cell[0] - r))
        dc = min(abs(cell[1] - c), env.GRID_SIZE - abs(cell[1] - c))
        dist = dr + dc
        if dist < best_dist:
            best_dist = dist
            if dr > dc:
                if (cell[0] - r) % env.GRID_SIZE > env.GRID_SIZE // 2:
                    move = 1
                else:
                    move = 2
            else:
                if (cell[1] - c) % env.GRID_SIZE > env.GRID_SIZE // 2:
                    move = 3
                else:
                    move = 4
    return [move, 0, 0]