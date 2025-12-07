def policy(env):
    # Strategy: Prioritize immediate matches to maximize score per move, then set up future matches.
    # Evaluate adjacent swaps to find the one with highest immediate match count, breaking ties by position.
    # Only act in IDLE state to avoid invalid actions during animations.
    if env.game_state != "IDLE":
        return [0, 0, 0]
    
    def evaluate_swap(grid, pos1, pos2):
        grid_copy = grid.copy()
        r1, c1 = pos1
        r2, c2 = pos2
        grid_copy[r1, c1], grid_copy[r2, c2] = grid_copy[r2, c2], grid_copy[r1, c1]
        return len(env._find_matches(grid_copy))
    
    if env.selected_gem is not None:
        sr, sc = env.selected_gem
        best_score = -1
        best_dir = 0
        for dr, dc, dir_val in [(-1,0,1), (1,0,2), (0,-1,3), (0,1,4)]:
            r2, c2 = sr + dr, sc + dc
            if 0 <= r2 < env.GRID_HEIGHT and 0 <= c2 < env.GRID_WIDTH:
                score = evaluate_swap(env.grid, (sr, sc), (r2, c2))
                if score > best_score:
                    best_score = score
                    best_dir = dir_val
        if best_score > 0:
            cr, cc = env.cursor_pos
            tr, tc = sr + (1 if best_dir==2 else -1 if best_dir==1 else 0), sc + (1 if best_dir==4 else -1 if best_dir==3 else 0)
            if cr == tr and cc == tc:
                return [0, 1, 0]
            move_dir = 3 if cc > tc else 4 if cc < tc else 1 if cr > tr else 2
            return [move_dir, 0, 0]
        else:
            return [0, 0, 1]
    
    best_score = -1
    best_swap = None
    for r in range(env.GRID_HEIGHT):
        for c in range(env.GRID_WIDTH):
            if c < env.GRID_WIDTH - 1:
                score = evaluate_swap(env.grid, (r, c), (r, c+1))
                if score > best_score:
                    best_score = score
                    best_swap = (r, c)
            if r < env.GRID_HEIGHT - 1:
                score = evaluate_swap(env.grid, (r, c), (r+1, c))
                if score > best_score:
                    best_score = score
                    best_swap = (r, c)
    
    if best_swap is None:
        return [0, 0, 0]
    
    tr, tc = best_swap
    cr, cc = env.cursor_pos
    if cr == tr and cc == tc:
        return [0, 1, 0]
    move_dir = 3 if cc > tc else 4 if cc < tc else 1 if cr > tr else 2
    return [move_dir, 0, 0]