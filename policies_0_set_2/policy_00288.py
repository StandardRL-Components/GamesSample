def policy(env):
    # This policy uses a greedy one-step lookahead to maximize matches with the target pattern.
    # It evaluates flipping each tile, chooses the one that maximizes matches (or minimizes distance if tied),
    # and moves the cursor there to flip. If no improving flip exists, it moves to the nearest mismatch.
    if env.game_over:
        return [0, 0, 0]
    
    current_matches = 0
    for r in range(4):
        for c in range(4):
            if env.grid[r, c] == env.target_grid[r, c]:
                current_matches += 1
                
    best_matches = current_matches
    best_pos = None
    best_dist = float('inf')
    for r in range(4):
        for c in range(4):
            delta = 0
            for dr, dc in [(0,0), (0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    if env.grid[nr, nc] == env.target_grid[nr, nc]:
                        delta -= 1
                    else:
                        delta += 1
            new_matches = current_matches + delta
            dist = abs(r - env.cursor_pos[0]) + abs(c - env.cursor_pos[1])
            if new_matches > best_matches or (new_matches == best_matches and dist < best_dist):
                best_matches = new_matches
                best_pos = (r, c)
                best_dist = dist
                
    if best_pos is not None and env.moves_left > 0:
        tr, tc = best_pos
        cr, cc = env.cursor_pos
        if cr == tr and cc == tc:
            return [0, 1, 0]
        if cr < tr:
            return [2, 0, 0]
        if cr > tr:
            return [1, 0, 0]
        if cc < tc:
            return [4, 0, 0]
        return [3, 0, 0]
        
    best_mismatch = None
    best_mismatch_dist = float('inf')
    for r in range(4):
        for c in range(4):
            if env.grid[r, c] != env.target_grid[r, c]:
                dist = abs(r - env.cursor_pos[0]) + abs(c - env.cursor_pos[1])
                if dist < best_mismatch_dist:
                    best_mismatch = (r, c)
                    best_mismatch_dist = dist
                    
    if best_mismatch is not None:
        tr, tc = best_mismatch
        cr, cc = env.cursor_pos
        if cr < tr:
            return [2, 0, 0]
        if cr > tr:
            return [1, 0, 0]
        if cc < tc:
            return [4, 0, 0]
        return [3, 0, 0]
        
    return [0, 0, 0]