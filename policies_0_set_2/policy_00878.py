def policy(env):
    """
    Maximizes reward by systematically scanning the grid for valid matches, prioritizing moves that create matches of 3+.
    Uses cursor movement to target optimal swaps, selects tiles when aligned, and swaps when adjacent to selected tile.
    Avoids invalid swaps by ensuring adjacency and recalculating best move each step based on current grid state.
    """
    GRID_SIZE = env.GRID_SIZE
    cursor_r, cursor_c = env.cursor_pos
    selected = env.selected_pos

    def find_matches(grid):
        matches = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == 0:
                    continue
                if c < GRID_SIZE - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                if r < GRID_SIZE - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def evaluate_swap(pos1, pos2):
        grid_copy = env.grid.copy()
        grid_copy[pos1], grid_copy[pos2] = grid_copy[pos2], grid_copy[pos1]
        return len(find_matches(grid_copy))

    best_score = -1
    best_swap = None
    grid = env.grid
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if c < GRID_SIZE - 1:
                score = evaluate_swap((r, c), (r, c+1))
                if score > best_score:
                    best_score = score
                    best_swap = ((r, c), (r, c+1))
            if r < GRID_SIZE - 1:
                score = evaluate_swap((r, c), (r+1, c))
                if score > best_score:
                    best_score = score
                    best_swap = ((r, c), (r+1, c))

    if best_swap is None:
        return [0, 0, 0]

    (r1, c1), (r2, c2) = best_swap

    if selected is None:
        target_r, target_c = r1, c1
        if cursor_r == target_r and cursor_c == target_c:
            return [0, 1, 0]
        else:
            dr = target_r - cursor_r
            dc = target_c - cursor_c
            if abs(dc) > abs(dr):
                return [4 if dc > 0 else 3, 0, 0]
            else:
                return [2 if dr > 0 else 1, 0, 0]
    else:
        if selected == (r1, c1):
            target_r, target_c = r2, c2
        elif selected == (r2, c2):
            target_r, target_c = r1, c1
        else:
            if cursor_r == selected[0] and cursor_c == selected[1]:
                return [0, 1, 0]
            else:
                dr = selected[0] - cursor_r
                dc = selected[1] - cursor_c
                if abs(dc) > abs(dr):
                    return [4 if dc > 0 else 3, 0, 0]
                else:
                    return [2 if dr > 0 else 1, 0, 0]

        if cursor_r == target_r and cursor_c == target_c:
            return [0, 1, 0]
        else:
            dr = target_r - cursor_r
            dc = target_c - cursor_c
            if abs(dc) > abs(dr):
                return [4 if dc > 0 else 3, 0, 0]
            else:
                return [2 if dr > 0 else 1, 0, 0]