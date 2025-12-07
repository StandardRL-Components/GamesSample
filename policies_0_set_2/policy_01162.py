def policy(env):
    # This policy maximizes reward by always selecting the move with the highest immediate match potential.
    # It evaluates all possible swaps, scores them based on match size and cascade potential,
    # moves the cursor to the best swap position, and executes the swap when ready.
    if env.animation_state != "IDLE":
        return [0, 0, 0]
    
    best_score = -1
    best_move = None
    grid = env.grid
    
    for r in range(env.GRID_HEIGHT):
        for c in range(env.GRID_WIDTH):
            # Check right swap
            if c < env.GRID_WIDTH - 1:
                grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c]
                matches = env._find_matches(grid)
                score = len(matches)
                if score >= 5:
                    score += 10
                if score > best_score:
                    best_score = score
                    best_move = ((r, c), 'right')
                grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c]
            
            # Check down swap
            if r < env.GRID_HEIGHT - 1:
                grid[r, c], grid[r+1, c] = grid[r+1, c], grid[r, c]
                matches = env._find_matches(grid)
                score = len(matches)
                if score >= 5:
                    score += 10
                if score > best_score:
                    best_score = score
                    best_move = ((r, c), 'down')
                grid[r, c], grid[r+1, c] = grid[r+1, c], grid[r, c]
    
    if best_move is None:
        return [0, 0, 0]
    
    (r, c), direction = best_move
    curr_c, curr_r = env.cursor_pos
    
    if curr_c == c and curr_r == r:
        if direction == 'right':
            return [0, 1, 0]
        else:
            return [0, 0, 1]
    else:
        if curr_c < c:
            return [4, 0, 0]
        elif curr_c > c:
            return [3, 0, 0]
        elif curr_r < r:
            return [2, 0, 0]
        else:
            return [1, 0, 0]