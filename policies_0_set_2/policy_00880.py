def policy(env):
    """
    Maximizes score by first checking for immediate matches from the current cursor position.
    If no immediate match, evaluates adjacent swaps to find the highest scoring move.
    Prioritizes moves that create matches, then moves cursor toward potential matches.
    Uses deterministic tie-breaking to avoid oscillation.
    """
    grid = env.grid
    cursor_r, cursor_c = env.cursor_pos
    selected = env.selected_crystal
    
    def find_matches(grid):
        matches = set()
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols - 2):
                if grid[r, c] > 0 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for r in range(rows - 2):
            for c in range(cols):
                if grid[r, c] > 0 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def evaluate_swap(r1, c1, r2, c2):
        if not (0 <= r2 < grid.shape[0] and 0 <= c2 < grid.shape[1]):
            return -1
        if grid[r1, c1] == 0 or grid[r2, c2] == 0:
            return -1
        new_grid = grid.copy()
        new_grid[r1, c1], new_grid[r2, c2] = new_grid[r2, c2], new_grid[r1, c1]
        return len(find_matches(new_grid))

    if selected is not None:
        sel_r, sel_c = selected
        best_score = -1
        best_dir = 0
        for dir, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)], 1):
            score = evaluate_swap(sel_r, sel_c, sel_r + dr, sel_c + dc)
            if score > best_score:
                best_score = score
                best_dir = dir
        if best_score > 0:
            return [best_dir, 0, 0]
        else:
            return [0, 0, 1]

    else:
        if grid[cursor_r, cursor_c] > 0:
            for dir, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)], 1):
                if evaluate_swap(cursor_r, cursor_c, cursor_r + dr, cursor_c + dc) > 0:
                    return [0, 1, 0]

        best_score = -1
        best_r, best_c = -1, -1
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    continue
                for dr, dc in [(0,1), (1,0)]:
                    r2, c2 = r + dr, c + dc
                    score = evaluate_swap(r, c, r2, c2)
                    if score > best_score:
                        best_score = score
                        best_r, best_c = r, c

        if best_score > 0:
            if cursor_r < best_r:
                return [2, 0, 0]
            elif cursor_r > best_r:
                return [1, 0, 0]
            elif cursor_c < best_c:
                return [4, 0, 0]
            elif cursor_c > best_c:
                return [3, 0, 0]

        directions = [4, 2, 3, 1]
        for d in directions:
            new_r = cursor_r + (1 if d == 2 else (-1 if d == 1 else 0))
            new_c = cursor_c + (1 if d == 4 else (-1 if d == 3 else 0))
            if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                return [d, 0, 0]

    return [0, 0, 0]