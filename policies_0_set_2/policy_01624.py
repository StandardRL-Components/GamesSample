def policy(env):
    # Strategy: For each selected robot, compute the shortest path to its exit using BFS, avoiding obstacles and other robots.
    # Move along the first step of this path. If no path exists or robot is rescued, switch to another robot.
    # This minimizes move penalties and maximizes rescue rewards by always taking optimal steps toward goals.
    def bfs_next_step(start, goal, obstacles_set):
        parent = {}
        queue = [start]
        parent[start] = None
        while queue:
            current = queue.pop(0)
            if current == goal:
                break
            x, y = current
            for dx, dy, dir_val in [(0,-1,1), (0,1,2), (-1,0,3), (1,0,4)]:
                nx, ny = x + dx, y + dy
                next_cell = (nx, ny)
                if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and
                    next_cell not in obstacles_set and next_cell not in parent):
                    parent[next_cell] = (current, dir_val)
                    queue.append(next_cell)
        if goal not in parent:
            return 0
        current = goal
        while parent[current][0] != start:
            current = parent[current][0]
        return parent[current][1]

    current_robot = env.robots[env.selected_robot_idx]
    if current_robot['rescued']:
        return [0, 1, 0]

    obstacles_set = {tuple(obs) for obs in env.obstacles}
    for robot in env.robots:
        if not robot['rescued'] and robot['id'] != current_robot['id']:
            obstacles_set.add(tuple(robot['pos']))
    
    exit_pos = env.exits[current_robot['id']]['pos']
    direction = bfs_next_step(tuple(current_robot['pos']), tuple(exit_pos), obstacles_set)
    if direction != 0:
        return [direction, 0, 0]
    else:
        return [0, 1, 0]