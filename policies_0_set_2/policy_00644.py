def policy(env):
    # This policy evaluates all possible moves by simulating their effects on pixel correctness and row/column completion.
    # It prioritizes moves that yield immediate positive rewards (correct placements) and completion bonuses while
    # avoiding moves that break correct pixels. If no beneficial move exists, it defaults to moving toward the nearest incorrect pixel.
    if env.game_over:
        return [0, 0, 0]
    if (env.pixel_grid == env.target_grid).all():
        return [0, 0, 0]
    
    x, y = env.selector_pos
    grid_size = env.GRID_SIZE
    current_grid = env.pixel_grid
    target_grid = env.target_grid
    
    best_score = -float('inf')
    best_action = 0
    
    directions = [1, 2, 3, 4]
    for d in directions:
        dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][d]
        nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
        
        new_grid = current_grid.copy()
        temp = new_grid[y, x].copy()
        new_grid[y, x] = new_grid[ny, nx]
        new_grid[ny, nx] = temp
        
        pixel_reward = 0
        old_correct_xy = (current_grid[y, x] == target_grid[y, x]).all()
        old_correct_nynx = (current_grid[ny, nx] == target_grid[ny, nx]).all()
        new_correct_xy = (new_grid[y, x] == target_grid[y, x]).all()
        new_correct_nynx = (new_grid[ny, nx] == target_grid[ny, nx]).all()
        
        if not old_correct_nynx and new_correct_xy:
            pixel_reward += 1.0
        if old_correct_nynx and not new_correct_xy:
            pixel_reward -= 0.2
        if not old_correct_xy and new_correct_nynx:
            pixel_reward += 1.0
        if old_correct_xy and not new_correct_nynx:
            pixel_reward -= 0.2
        
        completion_reward = 0
        for i in [y, ny]:
            was_complete = (current_grid[i, :] == target_grid[i, :]).all()
            now_complete = (new_grid[i, :] == target_grid[i, :]).all()
            if not was_complete and now_complete:
                completion_reward += 5
        for j in [x, nx]:
            was_complete = (current_grid[:, j] == target_grid[:, j]).all()
            now_complete = (new_grid[:, j] == target_grid[:, j]).all()
            if not was_complete and now_complete:
                completion_reward += 5
        
        total_reward = pixel_reward + completion_reward
        if total_reward > best_score:
            best_score = total_reward
            best_action = d
    
    if best_score > -float('inf'):
        return [best_action, 0, 0]
    
    min_dist = float('inf')
    fallback_action = 0
    for d in directions:
        dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][d]
        nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
        if not (current_grid[ny, nx] == target_grid[ny, nx]).all():
            dist = abs(nx - x) + abs(ny - y)
            if dist < min_dist:
                min_dist = dist
                fallback_action = d
    return [fallback_action, 0, 0]