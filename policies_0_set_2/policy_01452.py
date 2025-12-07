def policy(env):
    # Strategy: Maximize immediate rewards by scoring alignments of 3+ crystals, then prioritize columns with existing same-color crystals
    # to set up future wins. Avoid full columns and minimize unnecessary movement to reduce cooldown delays.
    
    def score_column(col):
        if env.grid[0, col] != 0:  # Column full
            return -9999
        # Find drop row
        row = env.GRID_HEIGHT - 1
        while row >= 0 and env.grid[row, col] != 0:
            row -= 1
        if row < 0:
            return -9999
        color = env.next_crystal_color_idx + 1
        score = 0
        # Check alignments in all directions
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # Includes the new crystal
            # Check positive direction
            for i in range(1, env.WIN_LENGTH):
                nx, ny = col + i*dx, row + i*dy
                if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT) or env.grid[ny, nx] != color:
                    break
                count += 1
            # Check negative direction
            for i in range(1, env.WIN_LENGTH):
                nx, ny = col - i*dx, row - i*dy
                if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT) or env.grid[ny, nx] != color:
                    break
                count += 1
            # Score alignment
            if count >= 5:
                score += 1000  # Winning move
            elif count == 4:
                score += 50
            elif count == 3:
                score += 10
        # Add small bonus for adjacent same-color crystals
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = col + dx, row + dy
                if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and env.grid[ny, nx] == color:
                    score += 1
        return score

    # Check if we can drop immediately in current column with good score
    current_col = env.selector_pos
    if env.drop_cooldown == 0 and env.grid[0, current_col] == 0:
        current_score = score_column(current_col)
        if current_score >= 10:  # Good immediate reward
            return [0, 1, 0]

    # Find best column to target
    best_score = -9999
    best_col = current_col
    for col in range(env.GRID_WIDTH):
        s = score_column(col)
        if s > best_score or (s == best_score and abs(col - current_col) < abs(best_col - current_col)):
            best_score = s
            best_col = col

    # Move toward best column or drop if already there
    if best_col == current_col:
        if env.drop_cooldown == 0 and env.grid[0, current_col] == 0:
            return [0, 1, 0]
        else:
            return [0, 0, 0]
    elif env.input_cooldown == 0:
        if best_col < current_col:
            return [3, 0, 0]
        else:
            return [4, 0, 0]
    else:
        return [0, 0, 0]