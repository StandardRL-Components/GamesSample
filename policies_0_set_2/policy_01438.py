def policy(env):
    # Strategy: Focus on placing towers along the path to maximize enemy coverage and damage.
    # Prioritize Cannon towers for high damage against stronger waves, use Slow towers for crowd control,
    # and Gatling for early waves. Always place when affordable on optimal path-adjacent cells.
    
    # Precompute path cells for coverage calculation
    path_cells = []
    for i in range(env.GRID_W):
        for j in range(env.GRID_H):
            if env.grid[j][i] == 1:
                path_cells.append((i, j))
    
    def compute_coverage(x, y, tower_idx):
        """Calculate number of path cells covered by a tower at (x,y)"""
        tower_def = env.TOWER_DEFS[tower_idx]
        range_sq = tower_def['range'] ** 2
        cell_center_x = (x + 0.5) * env.CELL_SIZE
        cell_center_y = (y + 0.5) * env.CELL_SIZE
        count = 0
        for (i, j) in path_cells:
            path_x = (i + 0.5) * env.CELL_SIZE
            path_y = (j + 0.5) * env.CELL_SIZE
            dx = cell_center_x - path_x
            dy = cell_center_y - path_y
            if dx*dx + dy*dy <= range_sq:
                count += 1
        return count

    # Determine optimal tower type based on wave and resources
    affordable_towers = [i for i, def_ in enumerate(env.TOWER_DEFS) if def_['cost'] <= env.resources]
    if not affordable_towers:
        return [0, 0, 0]  # Wait if cannot afford any tower
    
    if env.wave_number < 5:
        desired_tower = 0  # Gatling for early waves
    elif env.wave_number < 10:
        desired_tower = 2  # Slow Tower for mid waves
    else:
        desired_tower = 1  # Cannon for late waves
        
    # Cycle tower type if needed
    a2 = 1 if env.selected_tower_idx != desired_tower and desired_tower in affordable_towers else 0
    
    # Find best placement in 3x3 neighborhood
    best_score = -1
    best_pos = env.cursor_pos[:]
    cx, cy = env.cursor_pos
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < env.GRID_W and 0 <= y < env.GRID_H:
                if env.grid[y][x] == 0 and not any(t.grid_pos == (x, y) for t in env.towers):
                    score = compute_coverage(x, y, env.selected_tower_idx)
                    if score > best_score or (score == best_score and abs(dx) + abs(dy) < abs(best_pos[0]-cx) + abs(best_pos[1]-cy)):
                        best_score = score
                        best_pos = [x, y]
    
    # Move toward best position
    a0 = 0
    if best_pos != env.cursor_pos:
        if best_pos[0] > env.cursor_pos[0]:
            a0 = 4
        elif best_pos[0] < env.cursor_pos[0]:
            a0 = 3
        elif best_pos[1] > env.cursor_pos[1]:
            a0 = 2
        elif best_pos[1] < env.cursor_pos[1]:
            a0 = 1
    
    # Place tower if at optimal position and affordable
    a1 = 1 if (env.cursor_pos == best_pos and 
               env.resources >= env.TOWER_DEFS[env.selected_tower_idx]['cost'] and 
               env.grid[env.cursor_pos[1]][env.cursor_pos[0]] == 0 and
               not any(t.grid_pos == tuple(env.cursor_pos) for t in env.towers)) else 0

    return [a0, a1, a2]