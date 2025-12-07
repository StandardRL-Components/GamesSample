def policy(env):
    # Strategy: Maximize immediate reward by selecting the largest connected group of blocks, 
    # prioritizing groups with more than 10 blocks for bonus rewards. If no group is available, 
    # move toward the largest group to set up future clears, using Manhattan distance for efficient movement.
    if env.game_state != 'IDLE' or env.game_over:
        return [0, 0, 0]
    
    grid = env.grid
    cursor_x, cursor_y = env.cursor_pos
    
    # Check if current cursor position is part of a valid group (size >= 2)
    if grid[cursor_x, cursor_y] != -1:
        color = grid[cursor_x, cursor_y]
        queue = [(cursor_x, cursor_y)]
        visited = set([(cursor_x, cursor_y)])
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 16 and 0 <= ny < 10 and (nx, ny) not in visited and grid[nx, ny] == color:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        if len(visited) >= 2:
            return [0, 1, 0]
    
    # Find the largest group(s) in the grid
    visited_global = set()
    best_size = 0
    best_cells = []
    for x in range(16):
        for y in range(10):
            if (x, y) in visited_global or grid[x, y] == -1:
                continue
            color = grid[x, y]
            queue = [(x, y)]
            visited_local = set([(x, y)])
            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < 16 and 0 <= ny < 10 and (nx, ny) not in visited_local and grid[nx, ny] == color:
                        visited_local.add((nx, ny))
                        queue.append((nx, ny))
            visited_global.update(visited_local)
            size = len(visited_local)
            if size < 2:
                continue
            if size > best_size:
                best_size = size
                best_cells = list(visited_local)
            elif size == best_size:
                best_cells.extend(visited_local)
    
    if not best_cells:
        return [0, 0, 0]
    
    # Find closest cell in the largest group(s) to the cursor
    min_dist = float('inf')
    target_x, target_y = cursor_x, cursor_y
    for (x, y) in best_cells:
        dist = abs(x - cursor_x) + abs(y - cursor_y)
        if dist < min_dist:
            min_dist = dist
            target_x, target_y = x, y
    
    # Move toward target
    dx = target_x - cursor_x
    dy = target_y - cursor_y
    if abs(dx) > abs(dy):
        return [3 if dx < 0 else 4, 0, 0]
    else:
        return [1 if dy < 0 else 2, 0, 0]