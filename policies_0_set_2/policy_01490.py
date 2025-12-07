def policy(env):
    # Use BFS to find the shortest path to the exit, then take the first step in that path.
    # This minimizes steps and maximizes reward by reaching the exit quickly while avoiding walls.
    from collections import deque
    
    robot_pos = env.robot_pos
    exit_pos = env.exit_pos
    walls = env.walls
    grid_width = env.GRID_WIDTH
    grid_height = env.GRID_HEIGHT
    
    if robot_pos == exit_pos:
        return [0, 0, 0]
    
    queue = deque([(robot_pos, [])])
    visited = {robot_pos}
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    
    while queue:
        pos, path = queue.popleft()
        if pos == exit_pos:
            if path:
                first_step = path[0]
                dx = first_step[0] - robot_pos[0]
                dy = first_step[1] - robot_pos[1]
                for d in directions:
                    if (dx, dy) == (d[0], d[1]):
                        return [d[2], 0, 0]
            return [0, 0, 0]
        
        for dx, dy, a0 in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < grid_width and 
                0 <= new_pos[1] < grid_height and 
                new_pos not in walls and 
                new_pos not in visited):
                visited.add(new_pos)
                queue.append((new_pos, path + [new_pos]))
    
    for dx, dy, a0 in directions:
        new_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
        if (0 <= new_pos[0] < grid_width and 
            0 <= new_pos[1] < grid_height and 
            new_pos not in walls):
            return [a0, 0, 0]
    
    return [0, 0, 0]