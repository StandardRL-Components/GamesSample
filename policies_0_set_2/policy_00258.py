def policy(env):
    # Strategy: Prioritize building Cannons for high damage on path-adjacent cells, using Gatlings when resources are low.
    # Systematically scan grid for optimal placements near path turns and enemy choke points to maximize coverage and value.
    def is_buildable(x, y):
        if not (0 <= x < env.GRID_W and 0 <= y < env.GRID_H):
            return False
        if not env.buildable_grid[x, y]:
            return False
        for tower in env.towers:
            tx, ty = int(tower.pos.x // env.GRID_SIZE), int(tower.pos.y // env.GRID_SIZE)
            if tx == x and ty == y:
                return False
        return True

    def score_cell(x, y):
        if not is_buildable(x, y):
            return -1
        score = 0
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.GRID_W and 0 <= ny < env.GRID_H:
                if not env.buildable_grid[nx, ny]:
                    score += 10
        return score

    cx, cy = env.cursor_pos
    cannon_cost = env.tower_types[1]['cost']
    gatling_cost = env.tower_types[0]['cost']
    
    # Select tower type based on affordability and wave progression
    if env.current_wave >= 3 and env.resources >= cannon_cost and env.selected_tower_type != 1:
        return [0, 0, 1]
    elif env.resources >= gatling_cost and env.selected_tower_type != 0:
        return [0, 0, 1]
    
    # Build if current cell is high-value and affordable
    tower_cost = env.tower_types[env.selected_tower_type]['cost']
    if score_cell(cx, cy) > 0 and env.resources >= tower_cost and not env.prev_space_held:
        return [0, 1, 0]
    
    # Find best buildable cell near path turns
    best_score = -1
    best_cell = None
    for x in range(env.GRID_W):
        for y in range(env.GRID_H):
            s = score_cell(x, y)
            if s > best_score:
                best_score = s
                best_cell = (x, y)
    
    # Move toward best cell if found
    if best_cell:
        bx, by = best_cell
        if bx < cx and cx > 0:
            return [3, 0, 0]
        elif bx > cx and cx < env.GRID_W - 1:
            return [4, 0, 0]
        elif by < cy and cy > 0:
            return [1, 0, 0]
        elif by > cy and cy < env.GRID_H - 1:
            return [2, 0, 0]
    
    return [0, 0, 0]