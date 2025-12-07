def policy(env):
    # Use BFS to find the shortest path to the nearest gem, then take the first step in that path.
    # This maximizes reward by minimizing movement cost while prioritizing gem collection.
    from collections import deque
    
    if env.game_over or not env.gems:
        return [0, 0, 0]
    
    start = env.player_pos
    queue = deque([(start, [])])
    visited = {start}
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) in env.gems:
            if path:
                return [path[0], 0, 0]
            break
        
        for dx, dy, action in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < env.MAZE_WIDTH and 0 <= ny < env.MAZE_HEIGHT and 
                env.maze[ny, nx] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                new_path = path + [action] if path else [action]
                queue.append(((nx, ny), new_path))
    
    return [0, 0, 0]