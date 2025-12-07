def policy(env):
    # Strategy: Prioritize cells that complete rows/columns for bonus rewards. For each incorrect cell, compute potential reward (1 + 10*row_complete + 10*col_complete) minus movement cost. Choose highest-value target, then move efficiently using wrapped Manhattan distance.
    x, y = env.cursor_pos
    if env.player_grid[y, x] != env.target_grid[y, x] and env.moves_left > 0:
        if env.selected_color_idx == env.target_grid[y, x]:
            return [0, 1, 0]
        else:
            return [0, 0, 1]
    
    errors_per_row = np.sum(env.player_grid != env.target_grid, axis=1)
    errors_per_col = np.sum(env.player_grid != env.target_grid, axis=0)
    best_score = -1000
    target_x, target_y = x, y
    for gy in range(10):
        for gx in range(10):
            if env.player_grid[gy, gx] == env.target_grid[gy, gx]:
                continue
            row_bonus = 10 if errors_per_row[gy] == 1 else 0
            col_bonus = 10 if errors_per_col[gx] == 1 else 0
            reward = 1 + row_bonus + col_bonus
            dx = min(abs(gx - x), 10 - abs(gx - x))
            dy = min(abs(gy - y), 10 - abs(gy - y))
            dist = dx + dy
            score = reward - dist
            if score > best_score:
                best_score = score
                target_x, target_y = gx, gy
    
    dx = (target_x - x) % 10
    if dx > 5:
        dx -= 10
    dy = (target_y - y) % 10
    if dy > 5:
        dy -= 10
        
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    elif dy != 0:
        return [2 if dy > 0 else 1, 0, 0]
    return [0, 0, 0]