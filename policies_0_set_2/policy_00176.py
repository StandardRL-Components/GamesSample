def policy(env):
    # Strategy: Use BFS to find the shortest path to the nearest gem while avoiding traps.
    # This ensures efficient gem collection by always moving optimally towards the closest target.
    if env.game_over:
        return [0, 0, 0]
    
    obstacles = set(env.traps)
    targets = set(env.gems)
    if not targets:
        return [0, 0, 0]
    
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    start = env.player_pos
    queue = [(start, [])]
    visited = {start}
    
    while queue:
        (x, y), path = queue.pop(0)
        if (x, y) in targets:
            if path:
                return [path[0], 0, 0]
            return [0, 0, 0]
        
        for dx, dy, act in directions:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT:
                if pos not in obstacles and pos not in visited:
                    visited.add(pos)
                    queue.append((pos, path + [act]))
    
    for dx, dy, act in directions:
        nx, ny = start[0] + dx, start[1] + dy
        if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and (nx, ny) not in obstacles:
            return [act, 0, 0]
            
    return [0, 0, 0]