def policy(env):
    # Strategy: Use BFS to find the shortest path to the exit, avoiding pits. If no path exists, use Manhattan distance heuristics.
    # This maximizes reward by minimizing steps to exit while avoiding pit penalties and step costs.
    if env.game_over:
        return [0, 0, 0]
    
    current = env.robot_pos
    exit_pos = env.exit_pos
    if current == exit_pos:
        return [0, 0, 0]
    
    # BFS to find shortest path to exit
    queue = []
    queue.append((current[0], current[1], []))
    visited = set([current])
    dirs = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    
    found = False
    next_act = 0
    while queue:
        x, y, path = queue.pop(0)
        if (x, y) == exit_pos:
            if path:
                next_act = path[0]
            found = True
            break
        for dx, dy, act in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and (nx, ny) not in env.pits:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [act]
                    queue.append((nx, ny, new_path))
    
    if found:
        return [next_act, 0, 0]
    
    # Fallback: move in direction that minimizes Manhattan distance to exit
    best_action = 0
    best_dist = abs(current[0] - exit_pos[0]) + abs(current[1] - exit_pos[1])
    for dx, dy, act in dirs:
        nx, ny = current[0] + dx, current[1] + dy
        if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and (nx, ny) not in env.pits:
            dist = abs(nx - exit_pos[0]) + abs(ny - exit_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_action = act
    if best_action != 0:
        return [best_action, 0, 0]
    
    # Last resort: move to any adjacent non-pit cell
    for dx, dy, act in dirs:
        nx, ny = current[0] + dx, current[1] + dy
        if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and (nx, ny) not in env.pits:
            return [act, 0, 0]
    
    return [0, 0, 0]