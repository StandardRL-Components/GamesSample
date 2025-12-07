def policy(env):
    # Use BFS to find shortest path to exit, then take first step. This minimizes moves and maximizes score by reaching exit quickly.
    from collections import deque

    if env.player_pos == env.exit_pos:
        return [0, 0, 0]

    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    visited = set()
    queue = deque()
    visited.add(env.player_pos)
    queue.append((env.player_pos[0], env.player_pos[1], None))
    
    while queue:
        x, y, first_step = queue.popleft()
        if (x, y) == env.exit_pos:
            return [first_step, 0, 0] if first_step is not None else [0, 0, 0]
        
        for dx, dy, action in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and 
                env.maze[ny, nx] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                new_first_step = first_step if first_step is not None else action
                queue.append((nx, ny, new_first_step))
    
    for dx, dy, action in directions:
        nx, ny = env.player_pos[0] + dx, env.player_pos[1] + dy
        if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and 
            env.maze[ny, nx] == 0):
            return [action, 0, 0]
            
    return [0, 0, 0]