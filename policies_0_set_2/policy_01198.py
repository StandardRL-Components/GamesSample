def policy(env):
    """
    Maximizes reward by identifying the highest-scoring adjacent swap that creates matches and clears jellies.
    If no immediate swap is available from the current position, moves the selector toward the best available swap.
    Avoids invalid swaps and prioritizes matches that clear jellies for immediate reward.
    """
    def find_matches(grid, rows, cols):
        matches = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == -1:
                    continue
                if c <= cols - 3 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                if r <= rows - 3 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def evaluate_swap(r1, c1, r2, c2):
        if not (0 <= r2 < env.GRID_ROWS and 0 <= c2 < env.GRID_COLS):
            return -1
        grid_copy = env.grid.copy()
        grid_copy[r1, c1], grid_copy[r2, c2] = grid_copy[r2, c2], grid_copy[r1, c1]
        matches = find_matches(grid_copy, env.GRID_ROWS, env.GRID_COLS)
        if not matches:
            return -1
        score = len(matches)
        for (r, c) in matches:
            if env.jelly_grid[r, c] == 1:
                score += 0.5
        return score

    best_score = -1
    best_action = [0, 0, 0]
    current_r, current_c = env.selector_pos
    directions = [(0, 1, 4), (1, 0, 2), (0, -1, 3), (-1, 0, 1)]
    
    for dr, dc, a0_val in directions:
        r2, c2 = current_r + dr, current_c + dc
        if 0 <= r2 < env.GRID_ROWS and 0 <= c2 < env.GRID_COLS:
            score = evaluate_swap(current_r, current_c, r2, c2)
            if score > best_score:
                best_score = score
                best_action = [a0_val, 1, 0]
    
    if best_score > 0:
        return best_action
    
    best_global_score = -1
    target_r, target_c = current_r, current_c
    for r in range(env.GRID_ROWS):
        for c in range(env.GRID_COLS):
            for dr, dc, _ in [(0,1), (1,0)]:
                r2, c2 = r + dr, c + dc
                score = evaluate_swap(r, c, r2, c2)
                if score > best_global_score:
                    best_global_score = score
                    target_r, target_c = r, c
    
    if best_global_score <= 0:
        return [4, 0, 0]
    
    dr = (target_r - current_r) % env.GRID_ROWS
    if dr > env.GRID_ROWS // 2:
        dr -= env.GRID_ROWS
    dc = (target_c - current_c) % env.GRID_COLS
    if dc > env.GRID_COLS // 2:
        dc -= env.GRID_COLS
        
    if abs(dr) > abs(dc):
        a0 = 2 if dr > 0 else 1
    else:
        a0 = 4 if dc > 0 else 3
    return [a0, 0, 0]