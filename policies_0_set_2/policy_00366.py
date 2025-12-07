def policy(env):
    # Strategy: Prioritize blocking enemy paths near base to prevent damage, then build defensive maze walls.
    # For urgent threats (<5 steps from base), block their next move. Otherwise, build predefined wall pattern
    # around base to force longer enemy paths and maximize wave clear rewards.
    cursor = env.cursor_pos
    base = env.base_pos
    urgent_enemies = []
    for enemy in env.enemies:
        dist = abs(enemy['pos'][0] - base[0]) + abs(enemy['pos'][1] - base[1])
        if dist < 5:
            urgent_enemies.append(enemy)
    urgent_enemies.sort(key=lambda e: abs(e['pos'][0]-base[0]) + abs(e['pos'][1]-base[1]))
    
    target = None
    if urgent_enemies:
        enemy = urgent_enemies[0]
        next_step = env._pathfind_next_step(enemy['pos'], base)
        if next_step is not None:
            target_cell = (enemy['pos'][0] + next_step[0], enemy['pos'][1] + next_step[1])
            if (0 <= target_cell[0] < env.GRID_W and 0 <= target_cell[1] < env.GRID_H and
                env.grid[target_cell[1]][target_cell[0]] == env.CELL_EMPTY):
                target = target_cell
                
    if target is None:
        wall_plan = []
        for y in range(18, -1, -1):
            wall_plan.append((14, y))
            wall_plan.append((15, y))
            wall_plan.append((17, y))
            wall_plan.append((18, y))
        for pos in wall_plan:
            if env.grid[pos[1]][pos[0]] == env.CELL_EMPTY:
                target = pos
                break
                
    if target is not None:
        dx = target[0] - cursor[0]
        dy = target[1] - cursor[1]
        if dx == 0 and dy == 0:
            if env.grid[target[1]][target[0]] == env.CELL_EMPTY:
                return [0, 1, 0]
            else:
                return [0, 0, 0]
        if dx > 0:
            return [4, 0, 0]
        elif dx < 0:
            return [3, 0, 0]
        if dy > 0:
            return [2, 0, 0]
        elif dy < 0:
            return [1, 0, 0]
            
    return [0, 0, 0]