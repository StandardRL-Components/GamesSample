def policy(env):
    # Strategy: Maximize immediate reward by selecting the largest available group of adjacent blocks.
    # If current cursor position is part of a valid group (size>=2), select it. Otherwise move toward the largest group.
    if env.game_over:
        return [0, 0, 0]
    
    def find_group(x, y):
        color = env.grid[y, x]
        if color == 0:
            return []
        stack = [(x, y)]
        visited = set([(x, y)])
        group = []
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < env.GRID_SIZE and 0 <= ny < env.GRID_SIZE:
                    if (nx, ny) not in visited and env.grid[ny, nx] == color:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
        return group

    x0, y0 = env.cursor_pos
    current_group = find_group(x0, y0)
    if len(current_group) >= 2:
        return [0, 1, 0]
    
    visited = set()
    best_group = None
    best_size = 1
    best_min_dist = float('inf')
    for y in range(env.GRID_SIZE):
        for x in range(env.GRID_SIZE):
            if (x, y) in visited or env.grid[y, x] == 0:
                continue
            group = find_group(x, y)
            if len(group) < 2:
                continue
            for cell in group:
                visited.add(cell)
            min_dist = min(abs(cx - x0) + abs(cy - y0) for cx, cy in group)
            if len(group) > best_size or (len(group) == best_size and min_dist < best_min_dist):
                best_group = group
                best_size = len(group)
                best_min_dist = min_dist

    if best_group is None:
        return [0, 0, 0]
    
    target = min(best_group, key=lambda cell: abs(cell[0] - x0) + abs(cell[1] - y0))
    dx = target[0] - x0
    dy = target[1] - y0
    
    if dx > 0:
        movement = 4
    elif dx < 0:
        movement = 3
    elif dy > 0:
        movement = 2
    elif dy < 0:
        movement = 1
    else:
        movement = 0
        
    return [movement, 0, 0]