def policy(env):
    # Strategy: Use BFS to find the shortest path from the robot's current position to the exit.
    # This ensures optimal navigation through the maze by always taking the next step along the shortest path.
    # The other two actions are set to 0 since they are unused in the environment.
    from collections import deque

    if env.robot_pos == env.exit_pos:
        return [0, 0, 0]

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    action_map = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
    
    queue = deque([env.robot_pos])
    visited = {env.robot_pos: None}
    
    while queue:
        x, y = queue.popleft()
        if (x, y) == env.exit_pos:
            break
            
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.maze_dim and 0 <= ny < env.maze_dim and env.maze[ny, nx] == 0 and (nx, ny) not in visited:
                visited[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    
    if env.exit_pos not in visited:
        return [0, 0, 0]
    
    path = []
    current = env.exit_pos
    while current != env.robot_pos:
        path.append(current)
        current = visited[current]
    path.reverse()
    
    next_x, next_y = path[0]
    dx = next_x - env.robot_pos[0]
    dy = next_y - env.robot_pos[1]
    
    return [action_map[(dx, dy)], 0, 0]