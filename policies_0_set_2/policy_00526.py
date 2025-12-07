def policy(env):
    # Strategy: Prioritize moves that immediately clear lines for maximum reward, then set up future clears.
    # Evaluate each possible move by simulating the resulting grid state and scoring based on:
    # - Immediate line clears (highest priority, with bonus for multi-line clears)
    # - Potential for future clears (adjacent same-color blocks)
    # Avoid no-ops and break ties by movement order (up, down, left, right, none).
    
    if env.game_over:
        return [0, 0, 0]
    
    grid = env.grid
    y, x = env.player_pos
    best_action = [0, 0, 0]
    best_score = -1
    
    movements = [1, 2, 3, 4]  # up, down, left, right
    for move in movements:
        new_y, new_x = y, x
        if move == 1 and y > 0:
            new_y -= 1
        elif move == 2 and y < env.GRID_HEIGHT - 1:
            new_y += 1
        elif move == 3 and x > 0:
            new_x -= 1
        elif move == 4 and x < env.GRID_WIDTH - 1:
            new_x += 1
        else:
            continue
            
        color = grid[new_y, new_x]
        if color == -1:
            continue
            
        # Check horizontal line
        left = new_x
        while left >= 0 and grid[new_y, left] == color:
            left -= 1
        right = new_x
        while right < env.GRID_WIDTH and grid[new_y, right] == color:
            right += 1
        h_count = right - left - 1
        
        # Check vertical line
        up = new_y
        while up >= 0 and grid[up, new_x] == color:
            up -= 1
        down = new_y
        while down < env.GRID_HEIGHT and grid[down, new_x] == color:
            down += 1
        v_count = down - up - 1
        
        # Score immediate clears
        score = 0
        if h_count >= 3:
            score += 10 + h_count
        if v_count >= 3:
            score += 10 + v_count
            
        # Score potential future clears
        if score == 0:
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                adj_y, adj_x = new_y + dy, new_x + dx
                if (0 <= adj_y < env.GRID_HEIGHT and 0 <= adj_x < env.GRID_WIDTH and 
                    grid[adj_y, adj_x] == color):
                    score += 1
                    
        if score > best_score:
            best_score = score
            best_action = [move, 0, 0]
            
    return best_action