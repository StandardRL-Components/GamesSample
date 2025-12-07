def policy(env):
    # Strategy: Use BFS to find shortest path to exit, avoiding obstacles. 
    # Always take the first step of the shortest path to minimize distance and steps.
    # Secondary actions are unused in this environment, so set to 0.
    from collections import deque

    if env.robot_pos == env.exit_pos:
        return [0, 0, 0]

    grid = [[0 for _ in range(env.GRID_COLS)] for _ in range(env.GRID_ROWS)]
    for (x, y) in env.obstacles:
        grid[y][x] = 1

    queue = deque([env.robot_pos])
    visited = {env.robot_pos: None}
    found = False
    while queue and not found:
        x, y = queue.popleft()
        for dx, dy, action in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < env.GRID_COLS and 0 <= ny < env.GRID_ROWS and 
                grid[ny][nx] == 0 and (nx, ny) not in visited):
                visited[(nx, ny)] = (x, y)
                if (nx, ny) == env.exit_pos:
                    found = True
                    break
                queue.append((nx, ny))
        if found:
            break

    if not found:
        return [0, 0, 0]

    path = []
    current = env.exit_pos
    while current != env.robot_pos:
        path.append(current)
        current = visited[current]
    next_pos = path[-1]

    dx = next_pos[0] - env.robot_pos[0]
    dy = next_pos[1] - env.robot_pos[1]
    action_map = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
    return [action_map[(dx, dy)], 0, 0]