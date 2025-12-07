def policy(env):
    # Strategy: Place blocks adjacent to the core first to form a protective barrier, then expand outward.
    # Prioritize cells with existing adjacent blocks to create contiguous walls that maximize enemy deflections.
    # Move cursor efficiently to target cells using Manhattan distance, placing blocks when positioned correctly.
    if env.blocks_to_place <= 0:
        return [0, 0, 0]
    
    core_x, core_y = env.core_pos
    cursor_x, cursor_y = env.cursor_pos
    max_priority = -1
    target_x, target_y = None, None
    
    for y in range(env.GRID_HEIGHT):
        for x in range(env.GRID_WIDTH):
            if env.grid[y, x] != 0:
                continue
            dist = abs(x - core_x) + abs(y - core_y)
            priority = 0
            if dist == 1:
                priority = 1000
            else:
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and env.grid[ny, nx] > 0:
                        priority += 100
                priority += (env.GRID_WIDTH + env.GRID_HEIGHT - dist)
            
            if priority > max_priority or (priority == max_priority and (target_y is None or y < target_y or (y == target_y and x < target_x))):
                max_priority = priority
                target_x, target_y = x, y
    
    if target_x is None:
        return [0, 0, 0]
    
    if cursor_x == target_x and cursor_y == target_y:
        return [0, 1, 0]
    
    dx = target_x - cursor_x
    dy = target_y - cursor_y
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
    return [movement, 0, 0]