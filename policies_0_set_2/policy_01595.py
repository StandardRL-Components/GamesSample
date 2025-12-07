def policy(env):
    """Strategy: Use BFS to find the shortest path to the goal, avoiding obstacles. Return the first action of the path. If already at goal, do nothing."""
    robot = env.robot_pos
    goal = env.goal_pos
    if robot == goal:
        return [0, 0, 0]
    
    obstacles_set = set(tuple(obs) for obs in env.obstacle_pos)
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    visited = set()
    queue = []
    start = (robot[0], robot[1])
    visited.add(start)
    queue.append((robot[0], robot[1], None))
    
    while queue:
        x, y, first_action = queue.pop(0)
        for dx, dy, act in directions:
            nx, ny = x + dx, y + dy
            next_cell = (nx, ny)
            if next_cell in visited:
                continue
            if nx < 0 or nx >= env.GRID_SIZE or ny < 0 or ny >= env.GRID_SIZE:
                continue
            if next_cell in obstacles_set:
                continue
            if (nx, ny) == (goal[0], goal[1]):
                if first_action is None:
                    return [act, 0, 0]
                else:
                    return [first_action, 0, 0]
            visited.add(next_cell)
            if first_action is None:
                new_first_action = act
            else:
                new_first_action = first_action
            queue.append((nx, ny, new_first_action))
    
    for dx, dy, act in directions:
        nx, ny = robot[0] + dx, robot[1] + dy
        if 0 <= nx < env.GRID_SIZE and 0 <= ny < env.GRID_SIZE and (nx, ny) not in obstacles_set:
            return [act, 0, 0]
    
    return [0, 0, 0]