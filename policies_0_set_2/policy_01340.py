def policy(env):
    # Strategy: Maximize score by clearing largest groups first. Move cursor to nearest tile with >=3 matching neighbors, then activate space to clear. Avoid shift to prevent penalty.
    if env.game_over:
        return [0, 0, 0]
    
    cx, cy = env.cursor_pos
    grid = env.grid
    
    def get_group_size(x, y):
        if grid[x, y] == 0:
            return 0
        color = grid[x, y]
        visited = set()
        queue = [(x, y)]
        visited.add((x, y))
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT:
                    if (nx, ny) not in visited and grid[nx, ny] == color:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return len(visited)
    
    current_size = get_group_size(cx, cy)
    if current_size >= 3:
        return [0, 1, 0]
    
    best_dist = float('inf')
    best_tile = None
    for x in range(env.GRID_WIDTH):
        for y in range(env.GRID_HEIGHT):
            if grid[x, y] != 0 and get_group_size(x, y) >= 3:
                dist = abs(x - cx) + abs(y - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_tile = (x, y)
    
    if best_tile is not None:
        tx, ty = best_tile
        if abs(tx - cx) > abs(ty - cy):
            return [4 if tx > cx else 3, 0, 0]
        else:
            return [2 if ty > cy else 1, 0, 0]
    
    if cx < env.GRID_WIDTH - 1:
        return [4, 0, 0]
    elif cy < env.GRID_HEIGHT - 1:
        return [2, 0, 0]
    else:
        return [0, 0, 0]